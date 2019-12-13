"""
Simulation process.
From micro-level decision making and learning, to macro-level simulation of Users on a graph network.
"""
from typing import List
from queue import PriorityQueue
from event import CreateActorEvent, AccidentEvent
from graph import RoadGraph
from utils import MultimodalDistribution, get_time_from_traffic_distribution
from user import User, Personality, CommuteOutput
from DeepRL import DQNAgent
from actor import AbstractActor
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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

    def run(self, agent: DQNAgent):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []
        self.users = self.create_users(agent)

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        # Create the Simulation Actors
        event_queue = PriorityQueue()
        create_actor_events = self.create_actors_events(self.users)
        # print("users {}".format(len(self.users)))

        for ae in create_actor_events:
            event_queue.put_nowait(ae.get_priorized())
        
        # elements in form (time, event), to be ordered by first tuple member

        # Start Simulation
        while event_queue.qsize() > 0:
            _, event = event_queue.get_nowait()
            # print(event.at_time)
            new_events = event.act(self)
            for ev in new_events:
                # If event doesn't exceed max_run_time
                if ev.get_timestamp() < self.max_run_time:
                    event_queue.put_nowait(ev.get_priorized())

        # Set total_travel_time of all unfinished actors to max_run_time
        for a in self.actors:
            if not a.reached_dest():
                a.total_travel_time = self.max_run_time

        final_users = []

        for actor in self.actors:
            commute_out = CommuteOutput(actor.cost, actor.total_travel_time, actor.awareness, str(type(actor).__name__))
            user_info = dict()
            user_info["user"] = actor.user
            user_info["commute_output"] = commute_out
            user_info["utility"] = actor.user.calculate_utility_value(commute_out)
            final_users.append(user_info)

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            agent.update_replay_memory(
                (current_state, AbstractActor.convert_service[user_info["commute_output"].mean_transportation],
                    user_info["utility"], 1, True))
            agent.train(True, 1)

        # self.draw_graph()

    def create_actors_events(self,users: [User]) -> List[CreateActorEvent]:
        """Returns all scheduled CreateActorEvents"""
        return [
            CreateActorEvent(
                user.start_time, self.actor_constructor, user)
            for user in users
        ]

    def create_accident_events(self) -> List[AccidentEvent]:
        return [
            # AccidentEvent(10.0, (3, 6), 0.2)
        ]
    
    def create_users(self, agent: DQNAgent):
        users = []

        for _ in range(self.input_config["users"]["num_users"]):
            time = get_time_from_traffic_distribution(self.traffic_distribution)

            personality = Personality(1, 1, 1, True)
            user = User(personality, time)

            if np.random.random() > agent.epsilon:
                # Get action from Q table
                current_state = np.array(user.get_user_current_state())
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, agent.output_dim)

            mean_transportation = AbstractActor.convert_service.inverse[action][0]
            user.mean_transportation = mean_transportation
            users.append(user)

        return users
