import argparse
import itertools as it
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.ndimage import gaussian_filter1d
from sklearn.inspection import partial_dependence
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from smt.design_space import DesignSpace, FloatVariable
from smt.sampling_methods import LHS

# Disable warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Global Configuration (used by main and functions)
# -----------------------------------------------------------------------------
input_names_all = [
    "price_to_dispose",
    "scarcity",
    "density",
    "cluster_spread",
    "km_cost",
]

# Input names excluding 'density'
input_names = ["price_to_dispose", "scarcity", "cluster_spread", "km_cost"]

# -----------------------------------------------------------------------------
# Sobol and Annotation Utilities
# -----------------------------------------------------------------------------
def safe_sobol(Y_pred, problem):
    """Compute Sobol indices safely, skipping NaN vectors."""
    if np.isnan(Y_pred).all():
        return None
    if np.isnan(Y_pred).any():
        raise RuntimeError("Metamodel returned NaNs; cannot run Sobol analysis.")
    return sobol.analyze(problem, Y_pred, calc_second_order=True, print_to_console=False)


def sobol_to_df(Si, names):
    """Convert SALib Sobol output to a tidy DataFrame."""
    if Si is None:
        return pd.DataFrame(
            {
                "Parameter": names,
                "S1": [np.nan] * len(names),
                "S1_conf": [np.nan] * len(names),
                "ST": [np.nan] * len(names),
                "ST_conf": [np.nan] * len(names),
            }
        )

    df_s = pd.DataFrame(
        {
            "Parameter": names,
            "S1": Si["S1"],
            "S1_conf": Si.get("S1_conf", [np.nan] * len(names)),
            "ST": Si["ST"],
            "ST_conf": Si.get("ST_conf", [np.nan] * len(names)),
        }
    )
    df_s[["S1", "ST"]] = df_s[["S1", "ST"]].clip(lower=0.0)
    return df_s


def drop_variable(values, names, varname="density"):
    """Drop variable `varname` from both values and names."""
    mask = np.array(names) != varname
    return values[mask], [n for n in names if n != varname]


def annotate(ax, values, positions):
    """Annotate bar plot values (or mark as n/a if NaN)."""
    for val, xpos in zip(values, positions):
        if np.isnan(val):
            ax.annotate(
                "n/a",
                xy=(xpos, 0.02),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
            )
        else:
            ax.annotate(
                f"{val:.2f}",
                xy=(xpos, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

# -----------------------------------------------------------------------------
# Metamodeling Utilities
# -----------------------------------------------------------------------------
def eval_metamodel_or_nan(metamodel, param_ot):
    """Evaluate metamodel or return NaNs if metamodel is None."""
    if metamodel is None:
        return np.full(len(param_ot), np.nan)
    try:
        return np.array(metamodel(param_ot))[:, 0]
    except Exception:
        return np.array(metamodel.predict(np.array(param_ot)))


def build_pce_metamodel(X_np, Y_np, input_names, input_dist):
    """Build a Polynomial Chaos Expansion (PCE) metamodel."""
    if X_np.shape[0] < 5:
        raise RuntimeError("Too few samples to build a PCE.")

    X_ot = ot.Sample(X_np)
    Y_ot = ot.Sample(np.atleast_2d(Y_np).T)

    degree = 150
    enum = ot.HyperbolicAnisotropicEnumerateFunction(len(input_names), 0.999999)
    poly_coll = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * len(input_names), enum
    )
    trunc = ot.FixedStrategy(poly_coll, degree)
    proj = ot.LeastSquaresStrategy(X_ot, Y_ot)

    algo = ot.FunctionalChaosAlgorithm(X_ot, Y_ot, input_dist, trunc, proj)
    try:
        algo.setMaximumEvaluationNumber(int(1e6))
    except Exception:
        pass
    algo.run()
    chaos_result = algo.getResult()
    if not hasattr(chaos_result, "getMetaModel"):
        raise RuntimeError("PCE did not return a metamodel; check data/OpenTURNS.")
    return chaos_result.getMetaModel(), chaos_result


def all_sobol_indices(pce_result, input_names):
    """Compute all Sobol indices from a PCE metamodel."""
    sensitivity = ot.FunctionalChaosSobolIndices(pce_result)
    d = len(input_names)
    records = []

    # Loop over all subset sizes (1 to d)
    for order in range(1, d + 1):
        for idx in it.combinations(range(d), order):
            if order == 1:
                val = sensitivity.getSobolIndex(idx[0])
            elif order == 2:
                val = sensitivity.getSobolIndex([idx[0], idx[1]])
            else:
                val = sensitivity.getSobolIndex(ot.Indices(idx))
            vars_ = [input_names[i] for i in idx]
            records.append({"order": order, "variables": vars_, "Sobol_index": val})

    # Total-order indices
    for i in range(d):
        ST = sensitivity.getSobolTotalIndex(i)
        records.append({"order": "Total", "variables": [input_names[i]], "Sobol_index": ST})

    return pd.DataFrame(records)


def train_mlp(x_bin, y_bin, max_iter=2222):
    """Train an MLP model on the data for a specific 'bin'."""
    if len(x_bin) == 0:
         print("Warning: Skipping MLP training for an empty bin.")
         return None
         
    x_bin = MinMaxScaler().fit_transform(x_bin)
    mlp_bin = MLPRegressor(
        hidden_layer_sizes=(128, 50, 50, 32),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=0,
    )
    mlp_bin.fit(x_bin, y_bin)
    return mlp_bin

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def plot_sobol_per_bin(df_vlow_sobol, df_mid_sobol, df_high_sobol, target_name):
    """
    Plots Sobol indices (S1 and ST) per density bin.
    Indices for 'density' are omitted from the plot.
    """
    xplot = np.arange(len(input_names))
    width = 0.11

    # Extract and drop 'density' variable's indices
    s1_vlow, input_names_no_density = drop_variable(df_vlow_sobol["S1"].values, input_names_all)
    st_vlow, _ = drop_variable(df_vlow_sobol["ST"].values, input_names_all)
    s1_mid, _ = drop_variable(df_mid_sobol["S1"].values, input_names_all)
    st_mid, _ = drop_variable(df_mid_sobol["ST"].values, input_names_all)
    s1_high, _ = drop_variable(df_high_sobol["S1"].values, input_names_all)
    st_high, _ = drop_variable(df_high_sobol["ST"].values, input_names_all)

    colors = {
        "S1_vlow": "#F4A8FF",
        "ST_vlow": "#8B008B",
        "S1_mid": "#89CFF0",
        "ST_mid": "#1F4E79",
        "S1_high": "#7FFFD4",
        "ST_high": "#008B8B",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(xplot - 2.5 * width, s1_vlow, width, label="S1 (very_low)", color=colors["S1_vlow"])
    ax.bar(xplot - 1.5 * width, st_vlow, width, label="ST (very_low)", color=colors["ST_vlow"])
    ax.bar(xplot - 0.5 * width, s1_mid, width, label="S1 (mid)", color=colors["S1_mid"])
    ax.bar(xplot + 0.5 * width, st_mid, width, label="ST (mid)", color=colors["ST_mid"])
    ax.bar(xplot + 1.5 * width, s1_high, width, label="S1 (high)", color=colors["S1_high"])
    ax.bar(xplot + 2.5 * width, st_high, width, label="ST (high)", color=colors["ST_high"])

    annotate(ax, s1_vlow, xplot - 2.5 * width)
    annotate(ax, st_vlow, xplot - 1.5 * width)
    annotate(ax, s1_mid, xplot - 0.5 * width)
    annotate(ax, st_mid, xplot + 0.5 * width)
    annotate(ax, s1_high, xplot + 1.5 * width)
    annotate(ax, st_high, xplot + 2.5 * width)

    ax.set_xticks(xplot)
    ax.set_xticklabels(input_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sobol index")
    ax.set_title(f"Sobol indices (S1 & ST) across density bins for {target_name}")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_pdp_ice(mlp, X_lhs_df, target_name):
    """
    Plots Partial Dependence (PDP) and Individual Conditional Expectation (ICE) plots.
    The color of each ICE line is based on the 'density' variable.
    """
    features_to_plot = list(input_names)
    n_features = len(features_to_plot)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Normalization for the color bar (based on the density of the LHS dataset)
    norm = plt.Normalize(X_lhs_df["density"].min(), X_lhs_df["density"].max())
    cmap = plt.cm.jet

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]
        # X_lhs_df is already scaled [0, 1]
        pd_results = partial_dependence(mlp, X_lhs_df, [feat], kind="both", grid_resolution=100)
        grid = pd_results["grid_values"][0]
        pdp = pd_results["average"][0]
        # Smoothing for ICE
        ice = gaussian_filter1d(pd_results["individual"][0], sigma=3, axis=1)

        for j in range(ice.shape[0]):
            ax.plot(grid, ice[j, :], color=cmap(norm(X_lhs_df["density"].iloc[j])), alpha=0.5)

        ax.plot(grid, pdp, color="black", linewidth=5)
        ax.set_title(f"{target_name} PDP/ICE {feat}")
        ax.set_xlabel(feat) # Added x-label

    # Remove extra subplots if fewer than 4 features
    for k in range(n_features, 4):
        fig.delaxes(axes[k])

    # Add color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Density (Normalized)")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis on a dataset.")
    parser.add_argument("inputfile", type=str, help="Path to the input CSV file (e.g., data/Newbounds.csv).")
    parser.add_argument("--target", type=str, choices=["symbiosis", "price"], default="symbiosis", help="Target variable for analysis.")
    args = parser.parse_args()

    csv1 = args.inputfile
    target_name = args.target

    # -----------------------------------------------------------------------------
    # Load and Preprocess Data
    # -----------------------------------------------------------------------------
    try:
        df = pd.read_csv(csv1)
        # df = df[df["density"]<10**-2]
        print(df.shape)
    except FileNotFoundError:
        print(f"Error: Input file not found at {csv1}")
        return

    if target_name not in df.columns:
        print(f"Error: Target column '{target_name}' not found in '{csv1}'.")
        return

    df = df.groupby(input_names).mean().reset_index()

    # Transformations
    df["density"] = np.log10(df["density"])
    df["scarcity"] = np.log2(df["scarcity"])



    # -----------------------------------------------------------------------------
    # Design Space and Problem Definition
    # -----------------------------------------------------------------------------
    ds = DesignSpace(
        [
            FloatVariable(0, 200),
            FloatVariable(-2, 2),
            FloatVariable(-5, -1),
            FloatVariable(0, 0.5),
            FloatVariable(0, 2),
        ]
    )

    problem = {
        "num_vars": len(input_names_all),
        "names": input_names_all,
        "bounds": [[0, 1] for _ in input_names_all], # Bounds for scaled data
    }

    # Define density bins
    a,b = ds.get_x_limits()[2]
    bin_edges = [a+(b-a)/3,a+2*(b-a)/3]
    print(bin_edges)
    df_very_low = df[df["density"] < bin_edges[0]].reset_index(drop=True)
    df_mid = df[(df["density"] >= bin_edges[0]) & (df["density"] < bin_edges[1])].reset_index(drop=True)
    df_high = df[df["density"] >= bin_edges[1]].reset_index(drop=True)
    print(f"Total samples: {len(df)}")
    print(f"Samples (very_low): {len(df_very_low)}, (mid): {len(df_mid)}, (high): {len(df_high)}")


    # -----------------------------------------------------------------------------
    # Global Metamodeling (for PDP/ICE and Global Sobol)
    # -----------------------------------------------------------------------------
    x = df[input_names_all].values
    y = df[target_name].values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    marginals = [ot.Uniform(0, 1) for _ in input_names_all]
    input_dist = ot.ComposedDistribution(marginals)

    # 1. MLP (for PDP/ICE)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 50, 50, 32),
        activation="relu",
        solver="adam",
        max_iter=2222,
        random_state=0,
    )
    mlp.fit(x_scaled, y)
    print("Global MLP trained.")

    # 2. PCE (for Global Sobol)
    chaos_all = None
    try:
        _, chaos_all = build_pce_metamodel(x_scaled, y, input_names_all, input_dist)
        print("PCE metamodel constructed.")
        df_sobol = all_sobol_indices(chaos_all, input_names_all)
        df_sobol_sorted = df_sobol[df_sobol["order"] != "Total"].sort_values(
            by="Sobol_index", ascending=False
        ).reset_index(drop=True)
        print("\nSobol (all data - PCE):\n", df_sobol_sorted)
    except RuntimeError as e:
        print(f"PCE construction failed: {e}. Global Sobol analysis skipped.")


    # LHS sampling (for PDP/ICE)
    lhs = LHS(xlimits=np.array(ds.get_x_limits()), criterion="ese", random_state=42)
    X_lhs_unscaled = lhs(250)
    X_lhs_df = pd.DataFrame(MinMaxScaler().fit_transform(X_lhs_unscaled), columns=input_names_all)


    # -----------------------------------------------------------------------------
    # Metamodeling per Density Bin (for Sobol per bin)
    # -----------------------------------------------------------------------------
    # from IPython import embed; embed()
    metamodel_vlow = train_mlp(df_very_low[input_names_all].values, df_very_low[target_name].values, max_iter=2333)
    metamodel_mid = train_mlp(df_mid[input_names_all].values, df_mid[target_name].values)
    metamodel_high = train_mlp(df_high[input_names_all].values, df_high[target_name].values)
    print("MLPs per density bin trained.")

    # -----------------------------------------------------------------------------
    # Compute Sobol Indices per Bin (with Saltelli)
    # -----------------------------------------------------------------------------
    param_values = saltelli.sample(problem, 1024, calc_second_order=True)
    print("Saltelli sample shape:", param_values.shape)
    param_ot = ot.Sample(param_values.tolist())

    Y_vlow = eval_metamodel_or_nan(metamodel_vlow, param_ot)
    Y_mid = eval_metamodel_or_nan(metamodel_mid, param_ot)
    Y_high = eval_metamodel_or_nan(metamodel_high, param_ot)

    Si_vlow = safe_sobol(Y_vlow, problem)
    Si_mid = safe_sobol(Y_mid, problem)
    Si_high = safe_sobol(Y_high, problem)

    df_vlow_sobol = sobol_to_df(Si_vlow, input_names_all)
    df_mid_sobol = sobol_to_df(Si_mid, input_names_all)
    df_high_sobol = sobol_to_df(Si_high, input_names_all)

    print("\nSobol (very_low - MLP):\n", df_vlow_sobol)
    print("\nSobol (mid - MLP):\n", df_mid_sobol)
    print("\nSobol (high - MLP):\n", df_high_sobol)

    # -----------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------
    plot_sobol_per_bin(df_vlow_sobol, df_mid_sobol, df_high_sobol, target_name)
    plot_pdp_ice(mlp, X_lhs_df, target_name)


if __name__ == "__main__":
    main()