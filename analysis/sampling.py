import argparse
import numpy as np
import pandas as pd
import subprocess
from SALib.sample import latin, sobol
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil
import os

# Problem definition
problem = {
    'num_vars': 5,
    'names': ['price_to_dispose', 'log_2_scarcity', 'log_density', 'cluster_spread', 'km_cost'],
    'bounds': [
        [0, 200],     # price_to_dispose
        [-2, 2],      # log_2_scarcity 
        [-5, -1],     # log_density 
        [0, 0.5],     # cluster_spread
        [0, 2]       # km_cost
    ]
}

base_conf = """
nb_steps: 1000
nb_agents: 40
temperature_decay: 0.996
n_bins : 30
n_clusters : 4

processed_path: processed
visualization : no
animate : false
roads : false 
compute_regret: no 
"""

MASTIO_RUN_PATH = "MASTIO/run.py"

def run_simulation(index_params):
    """Run one simulation with given parameters, keeping index."""
    idx, params = index_params
    price_to_dispose, log_2_scarcity, log_density, cluster_spread, km_cost = params
    density = 10 ** log_density
    scarcity = 2 ** log_2_scarcity

    temp_dir = tempfile.mkdtemp()
    products_file = os.path.join(temp_dir, "products.csv")
    config_file = os.path.join(temp_dir, "bid_gen.yaml")

    try:
        import csv
        with open(products_file, mode="w", newline="") as prod_file:
            prod_writer = csv.writer(prod_file)
            prod_writer.writerow([
                "name", "sellers_rate", "buyers_rate",
                "market_price", "km_cost", "scarcity", "price_to_dispose"
            ])
            prod_writer.writerow(["test", 0.25, 0.75, 100,
                                  km_cost, scarcity, price_to_dispose])

        with open(config_file, mode="w") as conf_file:
            conf_file.write(f"products_path: {products_file}\n")
            conf_file.write(f"density: {format(density, '.8e')}\n")
            conf_file.write(f"cluster_spread: {cluster_spread}\n")
            conf_file.write(base_conf)

        result = subprocess.run(
            ["python", MASTIO_RUN_PATH, config_file],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Execution error:", result.stderr)
            return (idx, {
                "price_to_dispose": price_to_dispose,
                "scarcity": scarcity,
                "density": density,
                "cluster_spread": cluster_spread,
                "km_cost": km_cost,
                "price": np.nan,
                "symbiosis": np.nan
            })

        price, symbiosis = None, None
        for line in result.stdout.splitlines():
            if line.startswith("MEAN_PRICE_test="):
                price = float(line.split("=")[1])
            elif line.startswith("MEAN_symbiosis_test="):
                symbiosis = float(line.split("=")[1])

        return (idx, {
            "price_to_dispose": price_to_dispose,
            "scarcity": scarcity,
            "density": density,
            "cluster_spread": cluster_spread,
            "km_cost": km_cost,
            "price": price,
            "symbiosis": symbiosis
        })

    finally:
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024, help="Sample size N")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of repetitions per parameter set")
    parser.add_argument("output_file")
    parser.add_argument("--n_workers", type=int, default=21, help="Number of parallel workers")
    parser.add_argument("--method", choices=["lhs", "saltelli"], default="lhs",
                        help="Sampling method: lhs (Latin Hypercube) or saltelli (Sobol/Saltelli)")
    args = parser.parse_args()

    raw_csv = f"{args.output_file}"
    if os.path.exists(raw_csv):
        os.remove(raw_csv)

    # Sampling
    if args.method == "lhs":
        param_values = latin.sample(problem, args.n)
    elif args.method == "saltelli":
        param_values = sobol.sample(problem, args.n, calc_second_order=True)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Total parameter sets: {len(param_values)}")

    # Repeat each parameter set n_repeats times
    param_values_repeated = np.repeat(param_values, args.n_repeats, axis=0)
    print(f"Total simulations (with repeats): {len(param_values_repeated)}")

    indexed_params = list(enumerate(param_values_repeated))

    # Streaming ordered results
    next_index = 0
    buffer = {}  # stocke les résultats pas encore écrits

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor, \
         open(raw_csv, "w") as f_out:

        futures = [executor.submit(run_simulation, ip) for ip in indexed_params]

        header_written = False
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            idx, res = future.result()
            buffer[idx] = res

            # Écrire en ordre dès qu'on a le prochain index attendu
            while next_index in buffer:
                row = pd.DataFrame([buffer.pop(next_index)])
                row.to_csv(f_out, index=False, header=not header_written, mode="a")
                header_written = True
                next_index += 1

    print(f"Ordered streaming results saved to {raw_csv}")


if __name__ == "__main__":
    main()