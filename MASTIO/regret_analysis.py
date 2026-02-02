"""
This module provides tools for counter factual regret analysis. 
"""

import copy
from auction import Auction

class CounterfactualAnalyzer:
    """
    Analyzes counterfactual scenarios to calculate regret for agents.
    """
    def find_optimal_phi(self, target_agent, buyers, sellers, product_name):
        """
        Finds the optimal phi for a target agent by simulating the auction for each possible phi.

        Args:
            target_agent (Agent): The agent for whom to find the optimal phi.
            buyers (dict): A dictionary of buyer agents.
            sellers (dict): A dictionary of seller agents.
            product_name (str): The name of the product.

        Returns:
            tuple: A tuple containing the best phi and the maximum reward.
        """
        product_index = target_agent.product_indices[product_name]
        
        best_phi = None
        max_reward = -float('inf')

        # for test_phi in phis_to_test:
        for test_phi in target_agent.bins[product_index]:
            # Create deep copies of agents to avoid modifying the main simulation state
            sellers_copy = {sid: copy.deepcopy(s) for sid, s in sellers.items()}
            buyers_copy = copy.deepcopy(buyers)
            for s in sellers_copy.values():
                s.reset(reset_q_ini=False)
            for s in buyers_copy.values():
                s.reset(reset_q_ini=False)
            # Get the copied version of our target agent
            target_agent_copy = sellers_copy[target_agent.id]
            
            # Set the phi to the value we want to test
            target_agent_copy.phi[product_index] = test_phi
            
            # Create and run a temporary auction with the copied agents
            # We can reuse the other parameters as they are mostly for logging
            interactions = price_means = symbiosis_indicator = None
            temp_auction = Auction(buyers_copy, sellers_copy, product_name, interactions, price_means, symbiosis_indicator, record_stats=False)
            temp_auction.execute()
            
            # Calculate the reward for the target agent in this simulated auction
            q_to_sell = target_agent_copy.q_to_sell[product_index]
            sold_profit = target_agent_copy.sold_profit[product_index]
            price_to_dispose = target_agent_copy.price_to_dispose[product_index]
            
            reward = sold_profit - q_to_sell * price_to_dispose
            # print(test_phi, reward)
            
            if reward > max_reward:
                max_reward = reward
                best_phi = test_phi

        return best_phi, max_reward


    def calculate_regret(self, agent, optimal_reward, product_name):
        """
        Calculates the regret for a given agent.
        Regret = Optimal Reward - Actual Reward

        Args:
            agent (Agent): The agent for whom to calculate regret.
            optimal_reward (float): The optimal reward the agent could have achieved.
            product_name (str): The name of the product.

        Returns:
            float: The calculated regret.
        """
        product_index = agent.product_indices[product_name]
        actual_q_to_sell = agent.q_to_sell[product_index]
        actual_sold_profit = agent.sold_profit[product_index]
        price_to_dispose = agent.price_to_dispose[product_index]
        
        actual_reward = actual_sold_profit - actual_q_to_sell * price_to_dispose
        
        regret = optimal_reward - actual_reward
        # print("*************************")
        # print(agent.phi[product_index], actual_reward, regret)
        # print("*************************")
        return regret

    def calculate_total_seller_regret(self, sellers, buyers, product_name):
        """
        Calculates the total regret for all sellers for a given product.

        Args:
            sellers (dict): A dictionary of seller agents.
            buyers (dict): A dictionary of buyer agents.
            product_name (str): The name of the product.

        Returns:
            float: The total regret for all sellers.
        """
        total_regret = 0
        for seller_id, seller_agent in sellers.items():
            optimal_phi, optimal_reward = self.find_optimal_phi(
                seller_agent, buyers, sellers, product_name
            )
            individual_regret = self.calculate_regret(seller_agent, optimal_reward, product_name)
            total_regret += individual_regret
            # break # for one agent only
        return total_regret