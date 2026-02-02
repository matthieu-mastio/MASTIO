import argparse
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='sampling csv file')
    parser.add_argument("--target", choices=["price", "symbiosis"], default="symbiosis")
    args = parser.parse_args()

    problem = {
        'num_vars': 5,
        'names': ['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost'],
        'bounds': [
            [0, 200],
            [0.1, 2],
            [1e-5, 1e-1],
            [0, 1],
            [0.1, 1]
        ]
    }


    df_results = pd.read_csv(args.filename)
    df_mean = df_results.groupby(['price_to_dispose', 'scarcity', 'density', 'cluster_spread', 'km_cost']).mean().reset_index()
    df_mean['density'] = np.log10(df_mean['density'])


    Y = df_mean[args.target].values
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=True)


    x = np.arange(len(problem["names"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, Si["S1"], width, yerr=Si["S1_conf"], label='S1', color="#045c90")
    ax.bar(x + width/2, Si["ST"], width, yerr=Si["ST_conf"], label='ST', color="#7fadd1")
    ax.set_xticks(x)
    ax.set_xticklabels(problem["names"])
    ax.set_ylabel("Sobol index")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"sobol_bar.png")
    plt.show()
    plt.close()



    
    # Extraire S2 et les transformer en matrice
    n = problem['num_vars']
    S2_matrix = np.full((n, n), np.nan)

    # Remplir uniquement la partie inférieure de la matrice
    for i in range(n):
        for j in range(i):
            S2_matrix[i, j] = Si['S2'][j, i]

    # Visualisation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S2_matrix, cmap="Blues", vmin=0, vmax=np.nanmax(S2_matrix))

    # Ajouter les ticks et les labels (ticks majeurs)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(problem['names'], rotation=45, ha="right")
    ax.set_yticklabels(problem['names'])

    # Désactiver la visibilité des lignes du cadre (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # CORRECTIF : Utiliser les minor ticks pour les séparations entre les carrés
    # Placer les minor ticks aux frontières des cellules (-0.5, 0.5, 1.5, ...)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    # Dessiner la grille en utilisant les minor ticks
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5) # Largeur des séparations
    ax.tick_params(which="minor", bottom=False, left=False) # S'assurer que les minor ticks eux-mêmes ne sont pas visibles

    # Ajouter les valeurs numériques dans chaque cellule
    for i in range(n):
        for j in range(i): # Boucle pour n'afficher que la partie inférieure
            v = S2_matrix[i, j]
            text = ax.text(j, i, f"{v:.2f}",
        ha="center", va="center", color="white" if v > 0.2 else "black", fontsize=8)

    # Masquer les labels de la partie supérieure
    for i in range(n):
        for j in range(i + 1, n):
            ax.text(j, i, '', ha="center", va="center", color="black")

    # Ajouter colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("S2 index")

    ax.set_title("Second-order Sobol Indices (S2)")
    plt.tight_layout()
    plt.savefig("sobol_s2_heatmap.png", dpi=300)
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()
