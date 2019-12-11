"""
Simulation process.
From micro-level decision making and learning, to macro-level simulation of Users on a graph network.
"""
from typing import List
from queue import PriorityQueue
from event import CreateActorEvent, AccidentEvent
from graph import RoadGraph
from utils import MultimodalDistribution
import networkx as nx
import matplotlib.pyplot as plt

import random


class Simulator:
    """Runs a simulation from a given set of parameters"""

    def __init__(self,
                 config,
                 input_config,
                 actor_constructor,
                 stats_constructor,
                 traffic_distribution=MultimodalDistribution.default(),
                 seed=42):
        self.config = config
        self.input_config = input_config
        self.graph = RoadGraph(input_config)
        self.num_actors = config.num_actors
        self.actor_constructor = actor_constructor
        self.stats_constructor = stats_constructor
        self.traffic_distribution = traffic_distribution
        self.max_run_time = config.max_run_time
        self.stats = None
        self.actors = None

        random.seed(seed)
        # plt.ion()

    def draw_graph(self):

        
        #Trying to display graph
        # plt.ion()
        display_graph = self.graph.graph

        colors = [display_graph[u][v]['color'] for u, v in display_graph.edges]

        # weights = [display_graph[u][v]['weight']
        #            for u, v in display_graph.edges]

        print(display_graph.number_of_edges())

        pos = nx.spring_layout(display_graph)
        elarge = display_graph.edges(data=True)

        # nodes
        nx.draw_networkx_nodes(display_graph, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(display_graph, pos, edgelist=elarge,
                               width=6, edge_color=colors)

        # labels
        nx.draw_networkx_labels(display_graph, pos, edgelist=elarge,
                                width=6)
        print("vou desenhar o graph espero eu")
        plt.axis('off')

        display_graph.clear()
        plt.title("graph!!!")
        plt.draw()
        # plt.ioff()
        # plt.pause(0.01)
        plt.waitforbuttonpress(0)
        plt.clf()



    def run(self):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        # Create the Simulation Actors
        event_queue = PriorityQueue()
        for ae in self.create_actors_events() + self.create_accident_events():
            event_queue.put_nowait(ae.get_priorized())
        # elements in form (time, event), to be ordered by first tuple member

        # Start Simulation
        while event_queue.qsize() > 0:
            _, event = event_queue.get_nowait()
            new_events = event.act(self)
            for ev in new_events:
                # If event doesn't exceed max_run_time
                if ev.get_timestamp() < self.max_run_time:
                    event_queue.put_nowait(ev.get_priorized())

        # Set total_travel_time of all unfinished actors to max_run_time
        for a in self.actors:
            if not a.reached_dest():
                a.total_travel_time = self.max_run_time

        # self.draw_graph()
     

    def get_time_from_traffic_distribution(self) -> float:
        result = self.traffic_distribution()
        while not 0.0 < result < 24.0:
            result = self.traffic_distribution()
        return result

    def create_actors_events(self) -> List[CreateActorEvent]:
        """Returns all scheduled CreateActorEvents"""
        return [
            CreateActorEvent(
                self.get_time_from_traffic_distribution(), self.actor_constructor)
            for _ in range(self.num_actors)
        ]

    def create_accident_events(self) -> List[AccidentEvent]:
        return [
            # AccidentEvent(10.0, (3, 6), 0.2)
        ]
