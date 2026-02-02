import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Charger les données brutes
results_csv = sys.argv[1]
df = pd.read_csv(results_csv)

# Calculer la moyenne et l’écart type par groupe
grouped = df.groupby(["scarcity", "price_to_dispose", "density"])
summary = grouped.agg(
    mean_price=("price", "mean"),
    std_price=("price", "std"),
    mean_symbiose=("symbiose", "mean"),
    std_symbiose=("symbiose", "std")
).reset_index()

densities = sorted(summary["density"].unique())
colors = plt.cm.PuBu(np.linspace(0.5, 1, len(densities)))

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, scarcity in enumerate([0.5, 1, 2]):
    ax = axes1[i]
    subset = summary[summary["scarcity"] == scarcity]
    for k_idx, density in enumerate(densities):
        data_k = subset[subset["density"] == density].sort_values("price_to_dispose")
        ax.plot(
            data_k["price_to_dispose"],
            data_k["mean_price"],
            label=f"density= {density}",
            color=colors[k_idx],
            marker='o'
        )
        ax.fill_between(
            data_k["price_to_dispose"],
            data_k["mean_price"] - data_k["std_price"],
            data_k["mean_price"] + data_k["std_price"],
            color=colors[k_idx],
            alpha=0.3
        )
    ax.set_title(f"scarcity = {scarcity}")
    ax.set_xlabel("price_to_dispose")
    if i == 0:
        ax.set_ylabel("mean equilibrium price ± std")
    ax.grid(True)
    ax.legend(loc="lower left")
# plt.suptitle("Prix d'équilibre (moyenne ± écart type)")
plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, scarcity in enumerate([0.5, 1, 2]):
    ax = axes2[i]
    subset = summary[summary["scarcity"] == scarcity]
    for k_idx, density in enumerate(densities):
        data_k = subset[subset["density"] == density].sort_values("price_to_dispose")
        ax.plot(
            data_k["price_to_dispose"],
            data_k["mean_symbiose"],
            label=f"density= {density}",
            color=colors[k_idx],
            marker='s'
        )
        ax.fill_between(
            data_k["price_to_dispose"],
            data_k["mean_symbiose"] - data_k["std_symbiose"],
            data_k["mean_symbiose"] + data_k["std_symbiose"],
            color=colors[k_idx],
            alpha=0.3
        )
    ax.set_title(f"scarcity = {scarcity}")
    ax.set_xlabel("price_to_dispose")
    if i == 0:
        ax.set_ylabel("mean symbiose ± std")
    ax.grid(True)
    ax.legend(loc="lower right")
# plt.suptitle("Symbiose moyenne (moyenne ± écart type)")
plt.tight_layout()
plt.show()
