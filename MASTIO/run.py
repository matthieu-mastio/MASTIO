"""
This module provides the main entry point for running the simulation.
"""

from model import Model
import yaml
import subprocess
import argparse
from pathlib import Path

def run(params):
    """
    Runs the simulation with the given parameters.

    Args:
        params (dict): A dictionary of simulation parameters.
    """
    global model

    model = Model(params)
    image_files = model.start()
    if params.get("visualisation", False):
        print("Visualizations saved in:")
        for image in image_files:
            print("\t"+image)
        # subprocess.Popen(['eog', '--new-instance', image])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file', type=str)
    parser.add_argument('--scarcity', type=float)
    parser.add_argument('--price_to_dispose', type=int)
    parser.add_argument('--transport_cost', type=float)
    args = parser.parse_args()
    with open(args.conf_file, 'r') as file:
        params = yaml.safe_load(file)
    processed_path = params["processed_path"]
    Path(processed_path+"/images").mkdir(parents=True, exist_ok=True)
    Path(processed_path+"/regrets").mkdir(parents=True, exist_ok=True)
    run(params)
