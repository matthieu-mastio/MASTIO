"""
This module defines the Auction class, which manages the auction process for a single product.
"""
import random
import pandas as pd
import numpy as np
import os

os.environ['USE_PYGEOS'] = '0'

class Auction:
    """
    Manages the auction process for a single product between buyers and sellers.
    """


    def __init__(self, buyers, sellers, productName, interactions, price_means, symbiosis_indicator, record_stats=True):
        """
        Initializes an Auction.

        Args:
            buyers (dict): A dictionary of buyer agents.
            sellers (dict): A dictionary of seller agents.
            productName (str): The name of the product being auctioned.
            interactions (defaultdict): A dictionary to record interactions between agents.
            price_means (dict): A dictionary to record the mean price of the product.
            symbiosis_indicator (dict): A dictionary to record the symbiosis indicator.
            record_stats (bool, optional): Whether to record statistics. Defaults to True.
        """
        self.buyers = buyers
        self.sellers = sellers
        self.productName = productName
        self.interactions = interactions
        self.price_means = price_means
        self.symbiosis_indicator = symbiosis_indicator
        self.productIndice = next(iter(self.buyers.values())).product_indices[self.productName]
        self.record_stats = record_stats


    def compute_offers(self):
        """
        Computes the offers from all sellers to all buyers.

        Returns:
            pd.DataFrame: A DataFrame containing the offers.
        """
        offers_m = {}
        for seller in self.sellers.values():
            offers = []
            for buyer in self.buyers.values():
                offers.append(seller.seller_offer(buyer, self.productName))
            offers_m[seller.id] = offers 
        offers_m = pd.DataFrame.from_dict(offers_m, orient="index", columns=list(self.buyers.keys()))
        return offers_m


    def execute(self):
        """
        Executes the auction, matching buyers and sellers.
        """
        remaining_buyers = list(self.buyers.values())
        random.shuffle(remaining_buyers)
        remaining_sellers = list(self.sellers.values())
        for s in remaining_sellers:
            s.contractProposals = []

        offers_m = self.compute_offers()
        atLeastOneContract = True
        q_sold = 0
        weighted_price_sum = 0

        i = self.productIndice

        while atLeastOneContract and not offers_m.empty:
            atLeastOneContract = False

            # Buyers select offers
            for buyer in remaining_buyers.copy():
                q, p, seller_id = buyer.buyer_select_offer(offers_m[buyer.id], self.productName)
                if seller_id:
                    self.sellers[seller_id].seller_addContractProposal(q, p, buyer.id, self.productName)

            # Sellers choose and honor contracts
            for seller in remaining_sellers.copy():
                contracts = seller.seller_chooseContracts(self.productName)
                if contracts:
                    atLeastOneContract = True

                for buyer_id, q, p in contracts:
                    buyer = self.buyers[buyer_id]
                    buyer.buyer_honorContract(q, p, seller.id, self.productName)
                    seller.seller_honorContract(q, p, buyer.id, self.productName)

                    if self.record_stats:
                        self.interactions[(buyer.id, seller.id)] += q
                    q_sold += q
                    weighted_price_sum += q * p

                    if buyer.q_needed[i] <= 0:
                        if buyer in remaining_buyers:
                            remaining_buyers.remove(buyer)

                if seller.q_to_sell[i] <= 0:
                    if seller in remaining_sellers:
                        remaining_sellers.remove(seller)
                    offers_m = offers_m.drop(seller.id)

        if self.record_stats:
            if q_sold > 0:
                self.price_means[self.productName].append(weighted_price_sum / q_sold)
            else:
                if len(self.price_means[self.productName]) > 0:
                    last_price = self.price_means[self.productName][-1]
                    self.price_means[self.productName].append(last_price)
                else:
                    market_price = next(iter(self.buyers.values())).market_price[self.productIndice]
                    self.price_means[self.productName].append(market_price)

            q_bought = np.sum([buyer.q_needed_ini[i] - buyer.q_needed[i] for buyer in self.buyers.values()])
            q_needed = np.sum([buyer.q_needed_ini[i] for buyer in self.buyers.values()])
            q_to_sell = np.sum([seller.q_ini[i] for seller in self.sellers.values()])
            self.symbiosis_indicator[self.productName].append((q_bought/min(1,q_needed/q_to_sell))/q_to_sell)
            # self.symbiosis_indicator.append(2*q_bought/(q_needed+q_to_sell))
            # self.symbiosis_indicator.append(q_bought)
            # print("Rewards: ", [seller.r for seller in self.sellers.values()])
            # print("Phi: ", [seller.phi for seller in self.sellers.values()])
