from typing import List, Dict

from ipdb import set_trace

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_accumulated_actor_graph(actors_flow_acc: Dict[str, List[List[float]]], n_runs):
    """Plot the road network occupation during the simulation"""

    fig = plt.figure()


    ax1 = fig.add_subplot(111)


    print(actors_flow_acc.keys())
    for key in actors_flow_acc.keys():
        # Your x and y axis
        data = np.array(actors_flow_acc[key])
        x, y = data[:, 0], data[:, 1]
        # y = [y/n_runs, z/n_runs]
        y = y / n_runs

        # use a known color palette (see..)
    
        pal = sns.color_palette("Set1")
        ax1.plot(x, y, label=key, alpha=0.4)
    plt.xlabel("hours")
    plt.ylabel("actors in graph")
    plt.legend(loc='upper right')
    plt.xlim(right=30)
    plt.show()


def plot_accumulated_edges_graphs(edges_accumulated: Dict[str,Dict[str, List[List[float]]]], n_runs):
    """Plot the actors occupation of all edges during the simulation"""
    fig = plt.figure()
    n_edges = len(edges_accumulated.keys())
    edge_list = sorted(list(edges_accumulated.keys()))

    for i, e_key in enumerate(edge_list):
        for actor in edges_accumulated[e_key]:

            edge_data = edges_accumulated[e_key][actor]
            edge_data = np.array(edge_data)
            x, y = edge_data[:, 0], edge_data[:, 1]
            y = y / n_runs

            ax = fig.add_subplot(
                4,
                int((n_edges / 4 + 0.5)),
                i + 1
            )
            ax.text(.2, .9, str(e_key),
                    horizontalalignment='right',
                    transform=ax.transAxes)

            pal = sns.color_palette("Set1")
            ax.plot(x, y, label=actor, alpha=0.4)
        ax.legend(loc='upper right')
        plt.xlim(right=28)

    plt.show()
