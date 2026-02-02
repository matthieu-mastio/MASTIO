import subprocess
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from tqdm import tqdm 

scarcities = [0.5, 1, 2]
prices_to_dispose = range(0, 200, 20)
densities = [0.00001 , 0.0001, 0.001]

base_conf = """
nb_steps: 1000
nb_agents: 40
temperature_decay: 0.996
n_bins : 30
n_clusters : 1


products_path: /home/matt/git/MASTIO/data/products.csv 
processed_path: /home/matt/git/MASTIO/data/processed/square
visualization : no
animate : false
roads : false 
compute_regret: no 
"""

product_file = "/home/matt/git/MASTIO/data/products.csv"
configuration_file = "/home/matt/git/MASTIO/bid_gen.yaml"
results_csv = sys.argv[1]
nb_sim = 21

def run_simulation():
    command = ["python", "MASTIO/run.py", configuration_file]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Erreur d’exécution :", result.stderr)
        return None

    price, symbiose = None, None
    for line in result.stdout.splitlines():
        if line.startswith("MEAN_PRICE_test="):
            try:
                price = float(line.split("=")[1])
            except ValueError:
                print("Erreur parsing price:", line)
        elif line.startswith("MEAN_SYMBIOSE_test="):
            try:
                symbiose = float(line.split("=")[1])
            except ValueError:
                print("Erreur parsing symbiose:", line)

    if price is not None and symbiose is not None:
        return (price, symbiose)
    return None

total_runs = (
    len(scarcities) *
    len(prices_to_dispose) *
    len(densities) *
    nb_sim
)

file_exists = os.path.exists(results_csv)
with open(results_csv, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow([
            "scarcity", "price_to_dispose", "density",
            "run_id", "price", "symbiose"
        ])

    run_counter = 0
    with tqdm(total=total_runs, desc="Simulations", unit="run") as pbar:
        for scarcity in scarcities:
            for price_to_dispose in prices_to_dispose:
                for density in densities:
                    # print(f"Running: scarcity={scarcity}, price_to_dispose={price_to_dispose}, density={density}")

                    with open(product_file, mode="w", newline="") as prod_file:
                        prod_writer = csv.writer(prod_file)
                        prod_writer.writerow([
                            "name", "sellers_rate", "buyers_rate",
                            "market_price", "density", "scarcity", "price_to_dispose"
                        ])
                        prod_writer.writerow(["test", 0.25, 0.75, 100, 1, scarcity, price_to_dispose])

                    with open(configuration_file, mode="w", newline="") as conf_file:
                        conf_file.write(f"density: {format(density, '.8f')}\n")
                        conf_file.write(base_conf)

                    with ThreadPoolExecutor(max_workers=nb_sim) as executor:
                        futures = [executor.submit(run_simulation) for _ in range(nb_sim)]
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                price, symbiose = result
                                writer.writerow([
                                    scarcity, price_to_dispose, density,
                                    run_counter, price, symbiose
                                ])
                                run_counter += 1
                            pbar.update(1)  