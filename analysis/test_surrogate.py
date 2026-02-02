import numpy as np
import pandas as pd
import openturns as ot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import argparse

# --------------------------------------------------------------------------
# Configuration : variables d'entrée et target
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run sensitivity analysis on a dataset.")
parser.add_argument("inputfile", type=str, help="Path to the input CSV file (e.g., data/Newbounds.csv).")
parser.add_argument("--target", type=str, choices=["symbiosis", "price"], default="symbiosis", help="Target variable for analysis.")
args = parser.parse_args()
csv_file  = args.inputfile
target_name = args.target


input_names_all = ["price_to_dispose", "scarcity", "density", "cluster_spread", "km_cost"]

# --------------------------------------------------------------------------
# Chargement et preprocessing
# --------------------------------------------------------------------------
df = pd.read_csv(csv_file)

# Moyenne par combinaison d'inputs
df = df.groupby(input_names_all).mean().reset_index()

# Transformations
df["density"] = np.log10(df["density"])
df["scarcity"] = np.log2(df["scarcity"])

# Entrées / sorties
X = df[input_names_all].values
y = df[target_name].values

# Normalisation
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(y.reshape(-1,1))[:,0]

# --------------------------------------------------------------------------
# Jeu de test indépendant
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# --------------------------------------------------------------------------
# PCE : construction sur X_train
# --------------------------------------------------------------------------
# Distribution des variables uniformes [0,1]
marginals = [ot.Uniform(0, 1) for _ in input_names_all]
input_dist = ot.ComposedDistribution(marginals)

X_train_ot = ot.Sample(X_train)
Y_train_ot = ot.Sample(np.atleast_2d(y_train).T)

# Construction PCE
degree = 150  # ajustable selon ton dataset
enum = ot.HyperbolicAnisotropicEnumerateFunction(len(input_names_all), 0.999)
poly_coll = ot.OrthogonalProductPolynomialFactory(
    [ot.LegendreFactory()] * len(input_names_all), enum
)
trunc = ot.FixedStrategy(poly_coll, degree)
proj = ot.LeastSquaresStrategy(X_train_ot, Y_train_ot)

algo = ot.FunctionalChaosAlgorithm(X_train_ot, Y_train_ot, input_dist, trunc, proj)
algo.run()
chaos_train = algo.getResult().getMetaModel()

# --------------------------------------------------------------------------
# Évaluation PCE sur le jeu de test
# --------------------------------------------------------------------------
X_test_ot = ot.Sample(X_test)
Y_pred_ot = chaos_train(X_test_ot)
Y_pred = np.array(Y_pred_ot)[:, 0]

rmse_pce = np.sqrt(mean_squared_error(y_test, Y_pred))  # compatible toutes versions sklearn
r2_pce = r2_score(y_test, Y_pred)

print(f"PCE Test RMSE: {rmse_pce:.4f}")
print(f"PCE Test R²: {r2_pce:.4f}")

# Scatter plot
plt.scatter(y_test, Y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs réelles (test)")
plt.ylabel("Prédictions PCE")
plt.title("PCE : Prédiction vs Réalité (jeu test)")
plt.grid(True)
plt.show()

# --------------------------------------------------------------------------
# MLP : construction sur X_train
# --------------------------------------------------------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 50, 50, 32),
    activation="relu",
    solver="adam",
    max_iter=2222,
    random_state=0
)
mlp.fit(X_train, y_train)

# Évaluation MLP
y_pred_mlp = mlp.predict(X_test)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))  # compatible toutes versions
r2_mlp = r2_score(y_test, y_pred_mlp)

print(f"MLP Test RMSE: {rmse_mlp:.4f}")
print(f"MLP Test R²: {r2_mlp:.4f}")

# Scatter plot
plt.scatter(y_test, y_pred_mlp, alpha=0.6, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valeurs réelles (test)")
plt.ylabel("Prédictions MLP")
plt.title("MLP : Prédiction vs Réalité (jeu test)")
plt.grid(True)
plt.show()
