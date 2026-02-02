"""
This module defines the Agent class, which represents a participant in the market.
Agents can be buyers or sellers of products, and they learn to optimize their pricing strategies.
"""
import random
import numpy as np
from utils import norm_softmax

class Agent:
    """
    Represents an agent in the market simulation.

    Agents can be buyers or sellers of various products. They have a location,
    manage contract proposals, and learn to adjust their pricing strategies
    based on market feedback.
    """

    def __init__(self, id, location, product_names, temperature_decay=0.996, n_bins=30):
        """
        Initializes an Agent.

        Args:
            id (str): The unique identifier for the agent.
            location (tuple): The (x, y) coordinates of the agent.
            product_names (list): A list of names for the products in the market.
            temperature_decay (float, optional): The decay rate for the temperature
                in the Boltzmann exploration. Defaults to 0.996.
            n_bins (int, optional): The number of bins to use for discretizing
                the phi values. Defaults to 30.
        """
        self.id = id
        self.location = location
        self.contractProposals = []
        self.temperature_decay = temperature_decay
        self.distances = {}

        self.product_indices = {name: i for i, name in enumerate(product_names)}
        n = len(product_names)

        self.n = n
        self.n_bins = n_bins

        self.buy_scale = np.zeros(n)
        self.sell_scale = np.zeros(n)
        self.beta = np.full(n, np.nan)
        self.market_price = np.zeros(n)
        self.price_to_dispose = np.zeros(n)
        self.km_cost = np.zeros(n)
        self.sold_profit = np.zeros(n)
        self.phi = np.full(n, np.nan)
        self.temperature = np.ones(n)

        self.q_ini = np.zeros(n)
        self.q_to_sell = np.zeros(n)
        self.q_needed = np.zeros(n)
        self.q_needed_ini = np.zeros(n)

        self.bins = [None] * n
        self.bin_weights = [None] * n
        self.transport_costs = [None] * n


    def init_buyer(self, name, product):
        """
        Initializes the agent as a buyer for a specific product.

        Args:
            name (str): The name of the product.
            product (tuple): A tuple containing product information.
        """
        i = self.product_indices[name]
        _, _, market_price, _, _, _ = product
        self.buy_scale[i] = random.randint(1, 5)
        self.beta[i] = random.uniform(0.9, 1)
        self.market_price[i] = market_price


    def init_seller(self, name, product):
        """
        Initializes the agent as a seller for a specific product.

        Args:
            name (str): The name of the product.
            product (tuple): A tuple containing product information.
        """
        i = self.product_indices[name]
        sellers_rate, buyers_rate, market_price, km_cost, scarcity, price_to_dispose = product
        ratio_b_s = buyers_rate / sellers_rate
        self.sell_scale[i] = (ratio_b_s / scarcity) * random.randint(1, 5)
        self.market_price[i] = market_price
        self.price_to_dispose[i] = price_to_dispose
        self.km_cost[i] = km_cost
        self.sold_profit[i] = 0
        self.temperature[i] = 1


    def init_bins(self):
        """
        Initializes the bins for discretizing the phi values for each product.
        """
        for i in range(self.n):
            if self.sell_scale[i] > 0:
                phi_min = -self.price_to_dispose[i] / self.market_price[i]
                self.bins[i] = np.linspace(phi_min, 1, self.n_bins)
                self.bin_weights[i] = np.full(self.n_bins, np.nan)
                rand_bin = random.randint(0, len(self.bins[i])-1)
                self.phi[i] = self.bins[i][rand_bin]


    def reset(self, reset_q_ini=True):
        """
        Resets the agent's state for a new simulation step.

        Args:
            reset_q_ini (bool, optional): Whether to reset the initial quantities.
                Defaults to True.
        """
        for i in range(self.n):
            if self.buy_scale[i] > 0:
                if reset_q_ini:
                    q_needed_ini = max(1, int(self.buy_scale[i] * random.uniform(9, 11)))
                    self.q_needed_ini[i] = q_needed_ini
                self.q_needed[i] = self.q_needed_ini[i]
            if self.sell_scale[i] > 0:
                if reset_q_ini:
                    q_ini = max(1, int(self.sell_scale[i] * random.uniform(9, 11)))
                    self.q_ini[i] = q_ini
                self.q_to_sell[i] = self.q_ini[i]
                self.sold_profit[i] = 0


    def compute_transport_costs(self):
        """
        Computes the transport costs to all other agents.
        """
        for i in range(self.n):
            if self.sell_scale[i] > 0:
                km_cost = self.km_cost[i]
                self.transport_costs[i] = {a: d * km_cost for a, d in self.distances.items()}
                # self.transport_costs[i] = {a: d * km_cost / 1000 for a, d in self.distances.items()}


    def boltzmann_sampling(self, name):
        """
        Performs Boltzmann exploration to select a new phi value.

        Args:
            name (str): The name of the product.
        """
        # boltzmann exploration with thermal annealing
        i = self.product_indices[name]
        _, q_ini = self.q_to_sell[i], self.q_ini[i]
        bins = self.bins[i]
        bin_weights = self.bin_weights[i]
        price_to_dispose = self.price_to_dispose[i]
        temperature = self.temperature[i]

        r_min = q_ini * -price_to_dispose

        w = np.interp(np.arange(len(bin_weights)), np.flatnonzero(~np.isnan(bin_weights)), bin_weights[~np.isnan(bin_weights)])
        bin_probs = norm_softmax(w, temperature=temperature, val_min=r_min)
        chosen_bin = np.random.choice(self.n_bins, p=bin_probs)
        self.phi[i] = bins[chosen_bin]
        # print(f"{self.id} {name} exploring new chosen bin: {chosen_bin}")


    def ucb_sampling(self, name, c=2.0):
        """
        Performs Upper Confidence Bound (UCB) sampling to select a new phi value.

        Args:
            name (str): The name of the product.
            c (float, optional): The exploration parameter. Defaults to 2.0.
        """
        i = self.product_indices[name]
        bins = self.bins[i]
        weights = self.bin_weights[i]
        counts = self.bin_count[i]
        market_price = self.market_price[i]
        price_to_dispose = self.price_to_dispose[i]

        # Interpolate missing weights
        weights = np.interp(np.arange(len(weights)), np.flatnonzero(~np.isnan(weights)), weights[~np.isnan(weights)])

        total_visits = np.sum(counts) + 1  # avoid log(0)
        exploration_bonus = c * np.sqrt(np.log(total_visits) / (counts + 1e-5))
        ucb_values = weights + exploration_bonus

        chosen_bin = np.argmax(ucb_values)
        phi = np.random.uniform(bins[chosen_bin], bins[chosen_bin + 1])
        phi_min = -price_to_dispose / market_price
        self.phi[i] = max(phi_min, min(phi, 1))


    def learn(self, name):
        """
        Updates the agent's knowledge based on the outcome of the auction.

        Args:
            name (str): The name of the product.
        """
        i = self.product_indices[name]
        q_to_sell, _ = self.q_to_sell[i], self.q_ini[i]
        sold_profit = self.sold_profit[i]
        bins = self.bins[i]
        bin_weights = self.bin_weights[i]
        price_to_dispose = self.price_to_dispose[i]
        phi = self.phi[i]
        current_bin = np.digitize(phi, bins) - 1
        temperature = self.temperature[i]

        r = sold_profit - q_to_sell * price_to_dispose

        a = 0.4
        if np.isnan(bin_weights[current_bin]):
            bin_weights[current_bin] = r
        else:
            bin_weights[current_bin] = a * bin_weights[current_bin] + (1 - a) * r
        self.boltzmann_sampling(name)
        self.temperature[i] = temperature * self.temperature_decay


    def seller_offer(self, buyer, name):
        """
        Generates an offer for a buyer.

        Args:
            buyer (Agent): The buyer to make an offer to.
            name (str): The name of the product.

        Returns:
            tuple: A tuple containing the quantity and price of the offer.
        """
        i = self.product_indices[name]
        phi = self.phi[i]
        market_price = self.market_price[i]
        q_to_sell = self.q_to_sell[i]
        price = phi * market_price + self.transport_costs[i][buyer.id]
        return (q_to_sell, round(price, 2))


    def seller_addContractProposal(self, q, p, buyer_id, name):
        """
        Adds a contract proposal to the agent's list of proposals.

        Args:
            q (float): The quantity of the product.
            p (float): The price of the product.
            buyer_id (str): The ID of the buyer.
            name (str): The name of the product.
        """
        i = self.product_indices[name]
        benefit = q * (p - self.transport_costs[i][buyer_id])
        self.contractProposals.append((buyer_id, q, p, benefit))


    def seller_chooseContracts(self, name):
        """
        Chooses which contract proposals to accept.

        Args:
            name (str): The name of the product.

        Returns:
            list: A list of accepted contracts.
        """
        i = self.product_indices[name]
        self.contractProposals.sort(key=lambda x: x[3], reverse=True)
        acceptedContracts = []
        q_remaining = self.q_to_sell[i]
        for buyer_id, q, p, _ in self.contractProposals:
            if q < q_remaining:
                acceptedContracts.append((buyer_id, q, p))
                q_remaining -= q
            else:
                acceptedContracts.append((buyer_id, q_remaining, p))
                q_remaining = 0
                break
        return acceptedContracts


    def seller_honorContract(self, q, p, buyer_id, name):
        """
        Honors a contract by updating the agent's state.

        Args:
            q (float): The quantity of the product.
            p (float): The price of the product.
            buyer_id (str): The ID of the buyer.
            name (str): The name of the product.
        """
        i = self.product_indices[name]
        self.q_to_sell[i] -= q
        self.sold_profit[i] += q * (p - self.transport_costs[i][buyer_id])
        self.contractProposals = []
        # print(f"{self.id} sold {q} {name} at {p}$ to {buyer_id}, {self.q_to_sell[i]} to sell remaining")


    def buyer_select_offer(self, offers, name):
        """
        Selects the best offer from a list of offers.

        Args:
            offers (dict): A dictionary of offers from sellers.
            name (str): The name of the product.

        Returns:
            tuple: The best offer, or (0, 0, None) if no suitable offer is found.
        """
        i = self.product_indices[name]
        beta = self.beta[i]
        market_price = self.market_price[i]
        q_needed = self.q_needed[i]
        max_acceptable_price = beta * market_price
        min_offer = (0, float("inf"), None)
        for seller_id, offer in offers.items():
            _, p = offer
            if p < max_acceptable_price and p < min_offer[1]:
                min_offer = (q_needed, p, seller_id)
        return min_offer if min_offer[2] else (0, 0, None)


    def buyer_honorContract(self, q, p, seller_id, name):
        """
        Honors a contract by updating the agent's state.

        Args:
            q (float): The quantity of the product.
            p (float): The price of the product.
            seller_id (str): The ID of the seller.
            name (str): The name of the product.
        """
        i = self.product_indices[name]
        self.q_needed[i] -= q
        # print(f"{self.id} {name} bought {q} at {p}$ from {seller_id}, {self.q_needed[i]} needed remaining")