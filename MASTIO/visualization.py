"""
This module provides functions for visualizing simulation results.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('qt5Agg')


def plot_mean_r(reward_list, path="reward.png"):
    """
    Plots the mean reward over time.

    Args:
        reward_list (list): A list of mean reward values.
        path (str, optional): The path to save the plot. Defaults to "reward.png".

    Returns:
        str: The path to the saved plot.
    """
    plt.plot(reward_list)
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Time')
    plt.grid(True)
    plt.ylim(bottom=-1)
    plt.ylim(top=1)
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path


def plot_seller_phi(sellers, path="phi.png"):
    """
    Plots the evolution of phi for each seller.

    Args:
        sellers (dict): A dictionary of seller agents.
        path (str, optional): The path to save the plot. Defaults to "phi.png".

    Returns:
        str: The path to the saved plot.
    """
    # plt.figure(figsize=(10, 6))
    for seller_id, seller in sellers.items():
        plt.plot(seller.phi_hist, label=seller_id)
    plt.xlabel("Timestep")
    plt.ylabel("Phi")
    plt.title("Phi evolution")
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path


def plot_mean_p(price_list, market_price, price_to_dispose, path="price.png"):
    """
    Plots the mean price over time.

    Args:
        price_list (list): A list of mean price values.
        market_price (float): The market price of the product.
        price_to_dispose (float): The price to dispose of the product.
        path (str, optional): The path to save the plot. Defaults to "price.png".

    Returns:
        str: The path to the saved plot.
    """
    plt.plot(price_list, color="black")
    plt.axhline(y=-price_to_dispose, color='blue', linestyle='--', linewidth=1, label='disposal cost (30)')
    plt.axhline(y=market_price, color='red', linestyle='--', linewidth=1, label='Market price (100)')
    plt.text(len(price_list)*0.05, -price_to_dispose, 'Disposal Cost', color='blue', fontsize=12, verticalalignment='bottom')
    plt.text(len(price_list)*0.05, market_price, 'Market Price', color='red', fontsize=12, verticalalignment='bottom')
    plt.xlabel('Step')
    plt.ylabel('Mean Price')
    plt.title('Mean Price over Time')
    #plt.xlim(left=0)
    #plt.ylim(bottom=15)
    #plt.ylim(top=115)
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path


def smooth_curve(y, window_size=5):
    if len(y) < window_size:
        return y  # Pas assez de points pour lisser
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def plot_symbiosis_indicator(symbiosis_indicator, path="symbiosis.png", smooth=True, window_size=5):
    symbiosis_indicator = np.array(symbiosis_indicator)
    
    if smooth:
        y_smooth = smooth_curve(symbiosis_indicator, window_size=window_size)
        x = np.arange(len(symbiosis_indicator))
        x_smooth = x[len(x) - len(y_smooth):]
    else:
        y_smooth = symbiosis_indicator
        x_smooth = np.arange(len(symbiosis_indicator))

    plt.figure()
    plt.plot(x_smooth, y_smooth, label="Symbiosis Indicator (smoothed)" if smooth else "Symbiosis Indicator")
    plt.xlabel('Step')
    plt.ylabel('Symbiosis indicator')
    plt.title('Symbiosis indicator over Time')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path


def plot_agent_network(buyers, sellers, area, interaction_history, net, path="network.png"):
    """
    Plots the agent network with interactions.

    Args:
        buyers (dict): A dictionary of buyer agents.
        sellers (dict): A dictionary of seller agents.
        area (object): The simulation area object.
        interaction_history (list): A list of interaction dictionaries over time.
        net (object): The network object (if applicable).
        path (str, optional): The path to save the plot. Defaults to "network.png".

    Returns:
        str: The path to the saved plot.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    if hasattr(area, 'gdf'):
        gdf = area.gdf
        gdf.plot(ax=ax, color='lightgrey', edgecolor='white', zorder=1)
        ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
        ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])
    else:
        ax.set_xlim(0, area.width)
        ax.set_ylim(0, area.width)
        ax.add_patch(plt.Rectangle((0, 0), area.width, area.width, fill=False, edgecolor='black', lw=1))

    ax.set_title("Buyer-Seller Interaction (Static View)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Affiche les agents
    for buyer in buyers.values():
        ax.scatter(*buyer.location, c='blue', s=50, label='Buyer' if buyer.id == 'B0' else "", zorder=4)
    for seller in sellers.values():
        ax.scatter(*seller.location, c='red', s=50, marker='s', label='Seller' if seller.id == 'S0' else "", zorder=4)
    ax.legend()

    # Cumul des interactions pour chaque paire (sur tout l'historique)
    interaction = interaction_history[-1]
    max_interaction = max(interaction.values())

    # Dessine les lignes entre acheteurs et vendeurs
    for (buyer_id, seller_id), count in interaction.items():
        buyer = buyers[buyer_id]
        seller = sellers[seller_id]
        alpha = count / max_interaction
        ax.plot(
            [buyer.location[0], seller.location[0]],
            [buyer.location[1], seller.location[1]],
            color = 'black',
            linewidth = 1 + 3 * alpha,
            alpha = 0.1+0.9*alpha,
            zorder = 3
        )
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return(path)



def animate_agent_network(buyers, sellers, area, interaction_history, net, path="network.gif"):
    """
    Animates the agent network interactions over time.

    Args:
        buyers (dict): A dictionary of buyer agents.
        sellers (dict): A dictionary of seller agents.
        area (object): The simulation area object.
        interaction_history (list): A list of interaction dictionaries over time.
        net (object): The network object (if applicable).
        path (str, optional): The path to save the animation. Defaults to "network.gif".

    Returns:
        str: The path to the saved animation.
    """
    print("Computing animation...")
    fig, ax = plt.subplots(figsize=(15, 15))

    # Affiche le fond de carte (gdf)
    if hasattr(area, 'gdf'):
        gdf = area.gdf
        gdf.plot(ax=ax, color='lightgrey', edgecolor='white')
        ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
        ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])
    else:
        ax.set_xlim(0, area.width)
        ax.set_ylim(0, area.width)
        ax.add_patch(plt.Rectangle((0, 0), area.width, area.width, fill=False, edgecolor='black', lw=1))

    ax.set_title("Buyer-Seller Interaction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Draw agents once
    for buyer in buyers.values():
        ax.scatter(*buyer.location, c='blue', s=50, label='Buyer' if buyer.id == 'B0' else "")
    for seller in sellers.values():
        ax.scatter(*seller.location, c='red', s=50, marker='s', label='Seller' if seller.id == 'S0' else "")
    ax.legend()

    lines = []

    def update(frame):
        nonlocal lines
        for l in lines:
            l.remove()  # Remove previous lines
        lines = []

        interactions = interaction_history[frame]
        all_counts = [count for step in interaction_history for count in step.values()]
        max_interaction = max(all_counts) if all_counts else 1

        for (buyer_id, seller_id), count in interactions.items():
            buyer = buyers[buyer_id]
            seller = sellers[seller_id]
            alpha = count / max_interaction
            line = ax.plot(
                [buyer.location[0], seller.location[0]],
                [buyer.location[1], seller.location[1]],
                color='black',
                linewidth = 1 + 3 * alpha,
                alpha = 0.1 + 0.6 * alpha
            )[0]
            lines.append(line)

        ax.set_title(f"Step {frame}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(interaction_history), interval=200, repeat=False
    )
    ani.save(path)
    plt.close()
    return(path)

def plot_regret(regret_history, path="regret.png"):
    """
    Plots the regret of the analyzed agent over time.

    Args:
        regret_history (list): A list of regret values.
        path (str, optional): The path to save the plot. Defaults to "regret.png".

    Returns:
        str: The path to the saved plot.
    """
    plt.figure()
    plt.plot(regret_history, "o")
    plt.xlabel('Step')
    plt.ylabel('Regret')
    plt.title('Regret of the Analyzed Agent Over Time')
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path

def plot_regret_ma(data, window_size=10, path="regret_ma.png"):
    """
    Plots the moving average and standard deviation of a dataset.

    Args:
        data (list): Input data.
        window_size (int, optional): Moving average window size. Defaults to 10.
        path (str, optional): Path to save the plot. Defaults to "regret_ma.png".

    Returns:
        str: Path to the saved plot.
    """
    df = pd.DataFrame(data, columns=["Value"])
    df["Moving Average"] = df["Value"].rolling(window=window_size).mean()
    df["Moving Std"] = df["Value"].rolling(window=window_size).std()

    plt.figure(figsize=(10, 6))
    plt.plot(df["Moving Average"], label="Moving Average", color='blue', linewidth=2)

    plt.fill_between(df.index,
                     df["Moving Average"] - df["Moving Std"],
                     df["Moving Average"] + df["Moving Std"],
                     color='blue', alpha=0.2, label="±1 Std Dev")

    plt.title(f"Moving Average with ±1 Std Dev (Window = {window_size})")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path

    
def plot_regret_med(data, window_size=50, path="regret_med.png"):
    """
    Plots rolling median and rolling 25%-75% quantile range for a dataset.

    Args:
        data (list): Input data.
        window_size (int, optional): Rolling window size. Defaults to 10.
        path (str, optional): Path to save the plot. Defaults to "regret_med.png".

    Returns:
        str: Path to the saved plot.
    """
    df = pd.DataFrame(data, columns=["Value"])

    # Rolling median
    df["Rolling Median"] = df["Value"].rolling(window=window_size).median()
    df["Rolling Q25"] = df["Value"].rolling(window=window_size).quantile(0.25)
    df["Rolling Q75"] = df["Value"].rolling(window=window_size).quantile(0.75)

    plt.figure(figsize=(10, 6))
    plt.plot(df["Rolling Median"], label=f"Rolling Median (window={window_size})", color='blue', linewidth=2)

    plt.fill_between(df.index,
                     df["Rolling Q25"],
                     df["Rolling Q75"],
                     color='blue', alpha=0.2, label=f"Rolling 25%-75% Quantiles (window={window_size})")

    plt.title(f"Rolling Median with 25%-75% Quantiles (window={window_size})")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path