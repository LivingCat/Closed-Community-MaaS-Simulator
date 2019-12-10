"""
Statistics from simulation run.
Several metrics are updated as the simulation runs and then some analysis can be made.
"""
from typing import List, Tuple, DefaultDict
from matplotlib import pyplot as plt
from collections import defaultdict

import graph
import numpy as np

from actor import CarActor, BusActor, AbstractActor


class SimStats:

    save_path: str
    actors_in_graph: List[Tuple[float, int]]
    edges_flow_over_time: DefaultDict[Tuple[int, int], List[Tuple[float, int]]]
    # differencing between using atis or not
    # time, using atis, not using atis
    actors_atis: List[Tuple[float, int]]
    edges_flow_atis = DefaultDict[Tuple[int, int],
                                  List[Tuple[float,int]]]

    def __init__(self, g: graph.RoadGraph, save_path="data"):
        self.save_path = save_path
        self.graph = g
        self.actors = []
        self.actors_in_graph = [(0.0, 0)]
        self.edges_flow_over_time = defaultdict(lambda: [(0.0, 0)])
        self.actors_atis = defaultdict(lambda: [(0.0, 0)])
        # self.actors_atis = [(0.0, 0)]
        inner_default_dict = lambda: defaultdict(lambda: [(0.0, 0)])
        self.edges_flow_atis = defaultdict(inner_default_dict)
        pass

    def update_num_actors(self, actor: AbstractActor, ts: float, delta: int):
        """Update the number of actors in the network"""
        self.actors_in_graph.append((ts, delta + self.actors_in_graph[-1][1]))
        self.actors_atis[str(type(actor).__name__)].append((ts,
                                 delta))
        # self.actors_atis.append((ts, delta))


    def add_actor(self, actor:AbstractActor, ts: float):
        """Add an actor to the network"""
        self.update_num_actors(actor, ts, 1)

    def remove_actor(self, actor: AbstractActor, ts: float):
        """Remove an actor from the network"""
        self.update_num_actors(actor, ts, -1)

    def update_num_actors_edge(self, actor: AbstractActor, edge: Tuple[int, int], ts: float, delta: int):
        """Update the number of actors in a given edge"""
        self.edges_flow_over_time[edge].append(
            (ts, delta + self.edges_flow_over_time[edge][-1][1]))

        self.edges_flow_atis[edge][str(type(actor).__name__)].append((
            ts,
            delta))

    def add_actor_edge(self, actor: AbstractActor, ts: float, edge: Tuple[int, int]):
        """Add an actor to the given edge"""
        self.update_num_actors_edge(actor, edge, ts, 1)

    def remove_actor_edge(self,actor: AbstractActor, ts: float, edge: Tuple[int, int]):
        """Remove an actor from the given edge"""
        self.update_num_actors_edge(actor, edge, ts, -1)

    def add_actors(self, actors: List[AbstractActor]):
        """Store the actors present in the simulation"""
        self.actors = actors

    def plot(self):
        """Plotting system general usage"""
        data = np.array(self.actors_in_graph)
        x, y = data[:, 0], data[:, 1]
        plt.title("actors in system")
        plt.xlabel("hours")
        plt.ylabel("number of actors")
        plt.plot(x, y)
        plt.show()
