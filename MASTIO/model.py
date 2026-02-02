"""
This module defines the Model class, which orchestrates the entire simulation.
"""

from agent import Agent
import random
import pandas as pd
import numpy as np
from visualization import *
from utils import compute_od_matrix, uniform_cluster_centers
from collections import defaultdict
import os
os.environ['USE_PYGEOS'] = '0'
from area import SquareArea
import time
from auction import Auction
from regret_analysis import CounterfactualAnalyzer
import json

class Model:
    """
    Orchestrates the simulation, including agent creation, stepping through time, and visualization.
    """

    def __init__(self, params):
        """
        Initializes the Model.

        Args:
            params (dict): A dictionary of simulation parameters.
        """
        self.params = params
        self.processed_path = params["processed_path"]
        self.nb_steps = params.get("nb_steps", 1000)
        self.visualization = params.get("visualization", False)
        self.compute_regret = params.get("compute_regret", False)

        self.products_df = pd.read_csv(
            params["products_path"],
            dtype={"name": str, "sellers_rate": float, "buyers_rate": float, "market_price": float, "km_cost": float, "scarcity": float, "price_to_dispose": float}
        )
        self.products_df.set_index("name", inplace=True)
        self.product_names = list(self.products_df.index)
        self.product_indices = {name: i for i, name in enumerate(self.product_names)} 
        self.nb_products = len(self.product_names)
        self.tick = 0
        self.agents = {}
        self.nb_agents = params.get("nb_agents", 40)
        self.agent_to_analyze = None
        self.product_to_analyze = self.product_names[0] if self.compute_regret else None
        self.analyzer = None

        self.interactions = defaultdict(int)
        self.interaction_history = []
        self.reward_means = []
        self.price_means = {name: [] for name in self.product_names}
        self.symbiosis_indicator = {name: [] for name in self.product_names}
        self.regret_values = []

        width = np.sqrt(self.nb_agents/params.get("density", 0.1))
        self.area = SquareArea(width)
        self.create_rdm_agent()

        self.net = None

        self.distance_matrix = compute_od_matrix(self.agents, self.net)
        for agent in self.agents.values():
            agent.distances = self.distance_matrix.loc[agent.id]
            agent.compute_transport_costs()


    def create_rdm_agent(self):
        """
        Creates random agents for the simulation.
        Agents can be placed uniformly or in clusters.
        
        Args:
            params (dict): A dictionary of simulation parameters.
                Must contain:
                    - temperature_decay
                    - n_bins
                Optional:
                    - n_clusters (int): number of spatial clusters
        """
        print("Generating agents...")
        all_agents = []

        n_clusters = self.params.get("n_clusters", 4)
        cluster_spread = self.params.get("cluster_spread", 0.05) 
        cluster_std = self.area.width * cluster_spread


        # cluster_centers = [self.area.random_point() for _ in range(n_clusters)]
        cluster_centers = uniform_cluster_centers(self.area, n_clusters)

        for i in range(self.nb_agents):
            id = f"A{i}"

            if n_clusters > 1:
                cx, cy = random.choice(cluster_centers)
                while True:
                    x = np.random.normal(cx, cluster_std)
                    y = np.random.normal(cy, cluster_std)
                    if 0 <= x <= self.area.width and 0 <= y <= self.area.width:
                        location = (x, y)
                        break
            else:
                location = self.area.random_point()

            agent = Agent(
                id, 
                location, 
                self.product_names, 
                temperature_decay=self.params.get('temperature_decay', 0.996), 
                n_bins=self.params.get("n_bins", 30)
            )
            self.agents[id] = agent
            all_agents.append(agent)

        for name, product in self.products_df.iterrows():
            sellers_rate, buyers_rate, _, _, _, _ = product

            n_buyers = min(int(buyers_rate * self.nb_agents), len(all_agents))
            buyers = random.sample(all_agents, n_buyers)
            for agent in buyers:
                agent.init_buyer(name, product)

            remaining = [a for a in all_agents if a not in buyers]
            n_sellers = min(int(sellers_rate * self.nb_agents), len(remaining))
            sellers = random.sample(remaining, n_sellers)
            for agent in sellers:
                agent.init_seller(name, product)

            if self.compute_regret and self.analyzer is None:
                self.analyzer = CounterfactualAnalyzer()
                print("Calculating total regret for all sellers")

            print(f"Sellers for {name}: {[agent.id for agent in sellers]}")

        for agent in self.agents.values():
            agent.reset()
            agent.init_bins()


    def step(self):
        """
        Executes a single step of the simulation.
        """
        print(f"\nStep {self.tick} :")
        for name, _ in self.products_df.iterrows():
            i = self.product_indices[name]
            buyers = {}
            sellers = {}

            for agent in self.agents.values():
                if agent.buy_scale[i] > 0:
                    buyers[agent.id] = agent
                elif agent.sell_scale[i] > 0:
                    sellers[agent.id] = agent
                # TODO: gestion des décharges 

            auction = Auction(buyers, sellers, name, self.interactions, self.price_means, self.symbiosis_indicator)
            auction.execute()
            self.interaction_history.append(dict(self.interactions))

            if self.compute_regret and name == self.product_to_analyze:
                total_regret = self.analyzer.calculate_total_seller_regret(sellers, buyers, name)
                print(f"  Total seller regret ({name}): {total_regret:.2f}")
                self.regret_values.append(total_regret)

        for agent in sellers.values():
            agent.learn(name)
        for agent in self.agents.values():
            agent.reset()


    def start(self):
        """
        Starts the simulation and runs it for the configured number of steps.
        """
        start_time = time.time()
        while self.tick < self.nb_steps:
            self.step()
            self.tick += 1


        if self.compute_regret:
            print(f"\nRegret {self.product_to_analyze}: {[round(p, 2) for p in self.regret_values]}")
        for name in self.product_names:
            print(f"\nPrice {name}: {[round(p, 2) for p in self.price_means[name]]}")
            print(f"\nSymbiosis indicator {name}: {[round(p, 2) for p in self.symbiosis_indicator[name]]}")
            print(f"\nMEAN_PRICE_START_{name}={np.mean(self.price_means[name][:10])}")
            print(f"\nMEAN_PRICE_{name}={np.mean(self.price_means[name][-10:])}")
            print(f"\nMEAN_SYMBIOSE_{name}={np.mean(self.symbiosis_indicator[name][-10:])}")
        dist_no_diag = self.distance_matrix.mask(np.eye(len(self.distance_matrix), dtype=bool))
        min_distances = dist_no_diag.min(axis=1)
        mean_dist_min = min_distances.mean()
        print(f"\nSquare size : {self.area.width:.2f} km")
        print(f"\nMean of minimal distances between agents : {mean_dist_min:.2f} km")

        print(f"\nSimulation execution time: {round(time.time() - start_time, 2)}s\n")
        if self.visualization:
            return self.get_visualization()
        else:
            return []


    def get_visualization(self):
        """
        Generates and returns paths to visualization files.

        Returns:
            list: A list of file paths to the generated images.
        """
        image_files = []
        for name, product in self.products_df.iterrows():
            sellers_rate, buyers_rate, market_price, km_cost, scarcity, price_to_dispose = product
            density = self.params.get("density", 0.001)
            cluster_spread = self.params.get("cluster_spread", 1)
            price_means = self.price_means[name]
            symbiosis_indicator = self.symbiosis_indicator[name]
            file_id = f"{name}_{self.nb_agents}_{sellers_rate}_{buyers_rate}_{scarcity}_{km_cost}_{price_to_dispose}_{density}_{cluster_spread}"
            image_files.append(plot_symbiosis_indicator(symbiosis_indicator, path=f"{self.processed_path}/images/{file_id}_indicator.png"))
            image_files.append(plot_mean_p(price_means, market_price, price_to_dispose, path=f"{self.processed_path}/images/{file_id}_price.png"))

            i = self.product_indices[name]
            buyers = {aid : agent  for aid, agent in self.agents.items() if agent.buy_scale[i] > 0}
            sellers = {aid : agent  for aid, agent in self.agents.items() if agent.sell_scale[i] >0}
            image_files.append(plot_agent_network(buyers, sellers, self.area, self.interaction_history, self.net, path=f"{self.processed_path}/images/{file_id}_area.png"))

            if name == self.product_to_analyze and self.compute_regret and self.regret_values:
                regret_path=f"{self.processed_path}/regrets/{file_id}_regret.json"
                with open(regret_path, 'w') as f:
                    json.dump(self.regret_values, f)
                image_files.append(plot_regret(self.regret_values, path=f"{self.processed_path}/images/{file_id}_regret.png"))
                image_files.append(plot_regret_ma(self.regret_values, path=f"{self.processed_path}/images/{file_id}_regret_ma.png"))
                image_files.append(plot_regret_med(self.regret_values, path=f"{self.processed_path}/images/{file_id}_regret_med.png"))
            if self.params["animate"]:
                image_files.append(animate_agent_network(
                    buyers, sellers, self.area, self.interaction_history, self.net
                ))
        return image_files