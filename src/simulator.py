"""
Simulation process.
From micro-level decision making and learning, to macro-level simulation of Users on a graph network.
"""
from typing import List
from queue import PriorityQueue
from event import CreateActorEvent
from graph import RoadGraph
from utils import UnimodalDistribution, get_time_from_traffic_distribution, get_traffic_peaks
from user import User, Personality, CommuteOutput
from provider import Provider, STCP, Personal, Friends, Bicycle
from DeepRL import DQNAgent
from actor import Actor
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import csv

from parking_lot import ParkingLot


DEFAULT_VALUE_NUM = -100.0
DEFAULT_VALUE_STRING = ""
MIN_HOUSE_DISTANCE = 1
MAX_HOUSE_DISTANCE = 30
MAX_WAITING_TIME = 0.5

#Credits
N_CREDITS_DISCOUNT = 10
CREDIT_VALUE = 0.05
CREDIT_INCENTIVE = False
SOCIAL_CREDIT = False
SOCIAL_PERCENT = 0.1

#Agents indexes
AGENT_COST_INDEX = 0
AGENT_TIME_INDEX = 1
AGENT_ENVIRONMENT_INDEX = 2
AGENT_COMFORT_INDEX = 3

TIE_BREAK = "agent_cost"


class Simulator:
    """Runs a simulation from a given set of parameters"""

    def __init__(self,
                 config,
                 input_config,
                 actor_constructor,
                 providers,
                 stats_constructor,
                 run_ensamble,
                 run_agent_cluster_bool,
                 save_population,
                 import_population,
                 save_file,
                 traffic_distribution=UnimodalDistribution.default(),
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
        self.users = []
        self.providers = providers
        self.first_run = False
        self.runn = 0
        self.users_lost = dict(dict())
        self.save_population = save_population
        self.import_population = import_population
        self.save_file = save_file
        self.credits_used = {
            "car": 0,
            "bus": 0,
            "sharedCar": 0,
            "bike": 0,
            "walk": 0,
            "total": 0
        }
        self.credits_used_run = {
            "bus": 0,
            "sharedCar": 0,
            "total": 0
        }

        self.list_times = []
        self.list_cost = []

        self.distance_dict = dict()

        self.parking_lot = ParkingLot(200, 100, 0, 0)

        self.run_ensamble = run_ensamble
        self.run_agent_cluster_bool = run_agent_cluster_bool

        random.seed(seed)

    #Draw graph
    def draw_graph(self):
        nx.draw(self.graph.graph, pos=nx.planar_layout(self.graph.graph))

        plt.title("graph")
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.clf()

    #Calculates distance between every node in the graph
    def calculate_distance_nodes_destination(self):
        distance = []
        for node in range(0, int(self.input_config["graph"]["num_nodes"])):
            dist = nx.shortest_path_length(
                self.graph.graph, source=node, target=self.graph.nend)
            distance.append(dist)
        return distance

    #Creates friend relations for all the users
    def create_friends(self):
        friend_distrib_info = self.input_config["users"]["friends_distribution"]
        dist = getattr(scipy.stats, friend_distrib_info["distrib"])

        #Go through all the users of the system and input num_friends for each of them following the distribution
        for user in self.users:
            num_friend = int(dist.rvs(
                loc=friend_distrib_info["mean"], scale=friend_distrib_info["stand_div"]))
            if(num_friend > friend_distrib_info["max"]):
                user.num_friends = friend_distrib_info["max"]
            else:
                user.num_friends = num_friend

        for user in self.users:
            #use probabilities to select "friend category"
            #find users that fit that category
            #if no users fit into that category OR have remaining friends space than skip

            #friends categories are:
            # 0 - friends from same course and grade (50%)
            # 1 - friends from same course but different grades (+1 or -1) (20%)
            # 2 - friends from same grade but different courses (20%)
            # 3 - friends from different courses and grades (10%)
            friend_categories = [0, 1, 2, 3]
            friend_categories_probs = [0.5, 0.2, 0.2, 0.1]

            #categories
            cats = random.choices(
                friend_categories, weights=friend_categories_probs, k=user.num_friends)
            for cat in cats:
                if(cat == 0):
                    for poss_friend in self.users:
                       if(poss_friend != user and user.course == poss_friend.course and user.grade == poss_friend.grade and poss_friend.num_friends > 0 and (poss_friend not in user.friends)):
                           user.friends.append(poss_friend)
                           user.num_friends -= 1
                           poss_friend.friends.append(user)
                           poss_friend.num_friends -= 1
                           break
                elif(cat == 1):
                    for poss_friend in self.users:
                        grade_num = int(
                            user.grade[(user.grade.index("_") + 1)])
                        prev_grade_num = grade_num - 1
                        next_grade_num = grade_num + 1
                        prev_grade = "year_" + str(prev_grade_num)
                        next_grade = "year_" + str(next_grade_num)
                        if(poss_friend != user and user.course == poss_friend.course and (poss_friend.grade == prev_grade or poss_friend.grade == next_grade) and poss_friend.num_friends > 0 and (poss_friend not in user.friends)):
                            user.friends.append(poss_friend)
                            user.num_friends -= 1
                            poss_friend.friends.append(user)
                            poss_friend.num_friends -= 1
                            break
                elif(cat == 2):
                    for poss_friend in self.users:
                        if(poss_friend != user and user.course != poss_friend.course and poss_friend.grade == user.grade and poss_friend.num_friends > 0 and (poss_friend not in user.friends)):
                            user.friends.append(poss_friend)
                            user.num_friends -= 1
                            poss_friend.friends.append(user)
                            poss_friend.num_friends -= 1
                            break
                else:
                    for poss_friend in self.users:
                        if(poss_friend != user and user.course != poss_friend.course and poss_friend.grade != user.grade and poss_friend.num_friends > 0 and (poss_friend not in user.friends)):
                            user.friends.append(poss_friend)
                            user.num_friends -= 1
                            poss_friend.friends.append(user)
                            poss_friend.num_friends -= 1
                            break

        return True

    #Service selection when running mode is "Agent Cluster"
    #This mode means that we have a neural net responsible for each cluster
    #Neural net learns only with cluster members
    def choose_mode_agent_cluster(self, agents: List[DQNAgent]):

        agent_clusters_dict = {
            "cluster_0": 0,
            "cluster_1": 1,
            "cluster_2": 2,
            "cluster_3": 3,
            "cluster_4": 4
        }

        for user in self.users:
            cluster_index = agent_clusters_dict[user.cluster]
            agent = agents[cluster_index]

            while True:
                if random.random() > agent.epsilon:
                    # Get action from Q table
                    current_state = np.array(user.get_user_current_state())
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, agent.output_dim)

                provider = self.providers[action]
                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.has_private) and provider.name == "Personal"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #if the user chose to use "bike", as in, cycling but it doesnt own one, it gets a punishment
                elif((not user.has_bike) and provider.name == "Bicycle"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user owns a bicycle, wants to use it, but lives too far awar
                elif(user.has_bike and provider.name == "Bicycle" and not self.graph.check_has_route(user.house_node, "bike")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user wants to walk but lives too far away
                elif(provider.name == "Walking" and not self.graph.check_has_route(user.house_node, "walk")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    #Service selection when running mode is "ensemble"
    #This mode means that we have a learning agent for each of the parts of the utility function
    #Every learner votes on the service he thinks its best, according to the utility parameter he learns from
    def choose_mode_ensemble(self, agents: List[DQNAgent]):

        actions_dict = {
            "agent_cost": -1,
            "agent_time": -1,
            "agent_environmental": -1,
            "agent_comfort": -1
        }

        for user in self.users:
            #Users chooses action to take
            while True:
                #save the action each agent chose
                for index, agent in enumerate(agents):
                    if random.random() > agent.epsilon:
                        # Get action from Q table
                        current_state = np.array(user.get_user_current_state())
                        action = np.argmax(agent.get_qs(current_state))
                    else:
                        # Get random action
                        action = np.random.randint(0, agent.output_dim)
                    if(index == AGENT_COST_INDEX):
                        actions_dict["agent_cost"] = action
                    elif(index == AGENT_TIME_INDEX):
                        actions_dict["agent_time"] = action
                    elif(index == AGENT_ENVIRONMENT_INDEX):
                        actions_dict["agent_environmental"] = action
                    elif(index == AGENT_COMFORT_INDEX):
                        actions_dict["agent_comfort"] = action

                #chosen action for the user will be:
                # the majority action - if we have a majority
                # tied - in case of tie between actions we choose the one either the time agent or the cost agent chose
                # random

                count_action_dict = {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0
                }

                for agent_choice in actions_dict:
                    choice = actions_dict[agent_choice]
                    count_action_dict[str(choice)] += 1

                maxi = max(count_action_dict.values())
                indexes_max = [index for index in count_action_dict.keys(
                ) if count_action_dict[index] == maxi]

                #if there is only one action that received the most votes
                if(len(indexes_max) == 1):
                    provider = self.providers[int(indexes_max[0])]
                #we have a tie
                elif(len(indexes_max) == 2):
                    provider = self.providers[actions_dict[TIE_BREAK]]
                elif(len(indexes_max) > 2):
                    # Get random action
                    # print("random")
                    action = int(random.choice(indexes_max))
                    provider = self.providers[action]

                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.has_private) and provider.name == "Personal"):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                #if the user chose to use "bike", as in, cycling but it doesnt own one, it gets a punishment
                elif((not user.has_bike) and provider.name == "Bicycle"):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                #user owns a bicycle, wants to use it, but lives too far awar
                elif(user.has_bike and provider.name == "Bicycle" and not self.graph.check_has_route(user.house_node, "bike")):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                #user wants to walk but lives too far away
                elif(provider.name == "Walking" and not self.graph.check_has_route(user.house_node, "walk")):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    #Service selection when running mode is "descriptive"
    #This mode means that the users don't have learning capability and choose service based on the percentages from the survey
    #These percentages differ between clusters and are defined in the JSON file as "original choices"
    def choose_mode_descriptive(self):
        for user in self.users:
            chosen = False
            percents_cluster = self.input_config["users"]["clusters"][user.cluster]["original_choices"]

            for key in percents_cluster.keys():
                key_copy = str(key)
                limits = key_copy.split("-")
                limits = [int(limit) for limit in limits]
                keys = percents_cluster[key].keys()
                values = percents_cluster[key].values()

                if(len(limits) == 2):
                    if(user.distance_from_destination >= limits[0] and user.distance_from_destination < limits[1]):
                        mode = random.choices(list(keys), list(values), k=1)
                        for provider in self.providers:
                            if(provider.service == mode[0]):
                                user.mean_transportation = provider.service
                                user.provider = provider
                                chosen = True

                #upper bound
                if(len(limits) == 1):
                    if(user.distance_from_destination >= limits[0]):
                        mode = random.choices(list(keys), list(values), k=1)
                        for provider in self.providers:
                            if(provider.service == mode[0]):
                                user.mean_transportation = provider.service
                                user.provider = provider
                                chosen = True
                    else:
                        print("error")
                if(chosen):
                    break

    #Service selection when running on the default mode
    #Default mode has a single learner, using deep q-learning, that is responsible for all users

    def choose_mode(self, agent: DQNAgent):
        for user in self.users:
            #Users chooses action to take
            while True:
                if random.random() > agent.epsilon:
                    # Get action from Q table
                    current_state = np.array(user.get_user_current_state())
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, agent.output_dim)

                provider = self.providers[action]
                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.has_private) and provider.name == "Personal"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #if the user chose to use "bike", as in, cycling but it doesnt own one, it gets a punishment
                elif((not user.has_bike) and provider.name == "Bicycle"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user owns a bicycle, wants to use it, but lives too far awar
                elif(user.has_bike and provider.name == "Bicycle" and not self.graph.check_has_route(user.house_node, "bike")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user wants to walk but lives too far away
                elif(provider.name == "Walking" and not self.graph.check_has_route(user.house_node, "walk")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    #Service selection when running mode is "every agent nn"
    #This mode means that each of the users has their personal learner agent that applies deep q-learning (using a neural net)
    def choose_mode_every_agent_nn(self, agents):
        for user in self.users:
            #Users chooses action to take
            agent = agents[user.user_id-1]
            while True:
                if random.random() > agent.epsilon:
                    # Get action from Q table
                    current_state = np.array(user.get_user_current_state())
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, agent.output_dim)

                provider = self.providers[action]
                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.has_private) and provider.name == "Personal"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #if the user chose to use "bike", as in, cycling but it doesnt own one, it gets a punishment
                elif((not user.has_bike) and provider.name == "Bicycle"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user owns a bicycle, wants to use it, but lives too far awar
                elif(user.has_bike and provider.name == "Bicycle" and not self.graph.check_has_route(user.house_node, "bike")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user wants to walk but lives too far away
                elif(provider.name == "Walking" and not self.graph.check_has_route(user.house_node, "walk")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    #Service selection when running mode is "every agent ensemble"
    #This mode means that each users has a total of 4 learning agents, since each of them learns with a part of the utility function
    #Every learner votes on the service he thinks its best, according to the utility parameter he learns from
    def choose_mode_every_agent_ensemble(self, dict_agents):
        actions_dict = {
            "agent_cost": -1,
            "agent_time": -1,
            "agent_environmental": -1,
            "agent_comfort": -1
        }

        for user in self.users:
            #Users chooses action to take
            agents = dict_agents[user.user_id]
            while True:
                #save the action each agent chose
                for index, agent in enumerate(agents):
                    if random.random() > agent.epsilon:
                        # Get action from Q table
                        current_state = np.array(user.get_user_current_state())
                        action = np.argmax(agent.get_qs(current_state))
                    else:
                        # Get random action
                        action = np.random.randint(0, agent.output_dim)
                    if(index == AGENT_COST_INDEX):
                        actions_dict["agent_cost"] = action
                    elif(index == AGENT_TIME_INDEX):
                        actions_dict["agent_time"] = action
                    elif(index == AGENT_ENVIRONMENT_INDEX):
                        actions_dict["agent_environmental"] = action
                    elif(index == AGENT_COMFORT_INDEX):
                        actions_dict["agent_comfort"] = action

                #chosen action for the user will be:
                # the majority action - if we have a majority
                # tied - in case of tie between actions we choose the one either the time agent or the cost agent chose
                # random

                count_action_dict = {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0
                }

                for agent_choice in actions_dict:
                    choice = actions_dict[agent_choice]
                    count_action_dict[str(choice)] += 1

                maxi = max(count_action_dict.values())
                indexes_max = [index for index in count_action_dict.keys(
                ) if count_action_dict[index] == maxi]

                #if there is only one action that received the most votes
                if(len(indexes_max) == 1):
                    provider = self.providers[int(indexes_max[0])]
                #we have a tie
                elif(len(indexes_max) == 2):
                    provider = self.providers[actions_dict[TIE_BREAK]]
                elif(len(indexes_max) > 2):
                    # Get random action
                    action = int(random.choice(indexes_max))
                    provider = self.providers[action]

                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.has_private) and provider.name == "Personal"):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                #if the user chose to use "bike", as in, cycling but it doesnt own one, it gets a punishment
                elif((not user.has_bike) and provider.name == "Bicycle"):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                #user owns a bicycle, wants to use it, but lives too far awar
                elif(user.has_bike and provider.name == "Bicycle" and not self.graph.check_has_route(user.house_node, "bike")):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                #user wants to walk but lives too far away
                elif(provider.name == "Walking" and not self.graph.check_has_route(user.house_node, "walk")):
                    for agent in agents:
                        agent.update_replay_memory(
                            (user.get_user_current_state(), action,
                                self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                        agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    #Service selection when running mode is "every agent fast
    #This mode means that each users has one learning agent, that learns from the entiry utility function but, instead of using deep q-learning, uses the weighted average utility of the last 200 experiences
    def choose_mode_every_agent_fast(self, agents):
         for user in self.users:
            #Users chooses action to take
            agent = agents[user.user_id-1]
            while True:
                if random.random() > agent.epsilon:  # dizia np.random.random
                    # Get action from Q table
                    current_state = np.array(user.get_user_current_state())
                    action = agent.get_qs(current_state)
                else:
                    # Get random action
                    action = np.random.randint(0, agent.output_dim)

                provider = self.providers[action]
                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.has_private) and provider.name == "Personal"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #if the user chose to use "bike", as in, cycling but it doesnt own one, it gets a punishment
                elif((not user.has_bike) and provider.name == "Bicycle"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user owns a bicycle, wants to use it, but lives too far awar
                elif(user.has_bike and provider.name == "Bicycle" and not self.graph.check_has_route(user.house_node, "bike")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                #user wants to walk but lives too far away
                elif(provider.name == "Walking" and not self.graph.check_has_route(user.house_node, "walk")):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                            self.input_config["users"]["punishment_doesnt_have_mode"], 1, True))
                    agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    #Attribute house nodes to the users, according to the user's distance from destination

    def add_house_nodes(self):
        distances = self.calculate_distance_nodes_destination()
        for user in self.users:
            res_list = [i for i, value in enumerate(
                distances) if value == user.distance_from_destination]
            node = random.choice(res_list)
            user.add_house_node(node)

    #Sorting drivers function
    #Sorts drivers by their available seats
    def sort_drivers_fun(self, driver: User):
        return driver.available_seats

    #Sorting buses function
    #Sorts buses by the length of their route
    def sort_bus_func(self, bus_driver: User):
        return len(bus_driver.route)

    #Sorting actors function
    #Sorts actors by the time they arrived to the destination
    def sort_actors_fun(self, actor: Actor):
        return actor.traveled_nodes[-1][0]

    #Pick random rider user from a list
    def pick_random_rider(self, driver, users_want_ride_sharing, start_node):
         #start a new list for the possible riders
        possible_pickup = []
        poss_routes = self.graph.get_all_routes(
            start_node, driver.mean_transportation)
        all_nodes = [node for route in poss_routes for node in route]

        for rider in users_want_ride_sharing:
            if(rider.house_node in all_nodes and rider != driver and self.in_time_slot(rider, driver)):
                possible_pickup.append(rider)

        #There doesn't exist any other user that the driver can pick up
        if(len(possible_pickup) == 0):
            chosen_rider = ""
        else:
            chosen_rider = random.choice(possible_pickup)
        return (chosen_rider, possible_pickup)

    #Ride Sharing matching algorithm
    def ride_sharing_matching(self, users: List['User']):
        users_want_ride_sharing = users
        possible_drivers = [user for user in users if user.has_private]
        possible_drivers.sort(reverse=True, key=self.sort_drivers_fun)

        while(len(possible_drivers) > 0):
            driver = possible_drivers[0]
            if(driver.available_seats == 0):
                possible_drivers.remove(driver)
                continue
            pickup = []
            possible_pickup = []
            poss_routes = self.graph.get_all_routes(
                driver.house_node, driver.mean_transportation)
            all_nodes = [node for route in poss_routes for node in route]

            for rider in users_want_ride_sharing:
                if(rider.house_node in all_nodes and rider != driver and self.in_time_slot(rider, driver)):
                    possible_pickup.append(rider)

            #There doesn't exist a user that the driver can pick up
            if(len(possible_pickup) == 0):
                possible_drivers.remove(driver)
                continue

            chosen_rider = random.choice(possible_pickup)
            pickup.append(chosen_rider)

            #remove chosen rider from list of users who want to ride share
            users_want_ride_sharing.remove(chosen_rider)

            #remove pickups which belong to the possible drivers
            if(chosen_rider in possible_drivers):
                possible_drivers.remove(chosen_rider)

            possible_pickup.remove(chosen_rider)
            driver.available_seats -= 1

            while (driver.available_seats > 0 and len(possible_pickup) > 0):
                #moves to the next node
                #checks the users it can pickup from the new node
                #start a new list for the possible riders
                possible_pickup = []
                new_node = pickup[-1].house_node
                poss_routes = self.graph.get_all_routes(
                    new_node, driver.mean_transportation)
                all_nodes = [node for route in poss_routes for node in route]

                for rider in users_want_ride_sharing:
                    if(rider.house_node in all_nodes and rider != driver and self.in_time_slot(rider, driver)):
                        possible_pickup.append(rider)

                #There doesn't exist any other user that the driver can pick up
                if(len(possible_pickup) == 0):
                    break

                chosen_rider = random.choice(possible_pickup)
                pickup.append(chosen_rider)

                #remove chosen rider from list of users who want to ride share
                users_want_ride_sharing.remove(chosen_rider)

                #remove pickups which belonged to the possible drivers
                if(chosen_rider in possible_drivers):
                    possible_drivers.remove(chosen_rider)

                possible_pickup.remove(chosen_rider)
                driver.available_seats -= 1
            driver.users_to_pick_up = pickup
            #remove driver from possible drivers, since the matching for him has ended
            possible_drivers.remove(driver)
            #remove driver from users that want ride sharing, since he has been chosen to be the driver and has been matched
            users_want_ride_sharing.remove(driver)
        return users_want_ride_sharing

    #Check if the time difference between driver and rider start time is within an acceptable amount
    def in_time_slot(self, rider: User, driver: User):
        wait_time = MAX_WAITING_TIME * rider.personality.willingness_to_wait
        return (abs(driver.start_time - rider.start_time) <= wait_time)

    #Public Transport Matching algorithm

    def public_transport_matching(self, users: List['User'], bus_users: List['User']):

        users_want_public_transport = users

        users_not_matched = []

        bus_drivers = bus_users

        bus_drivers.sort(key=self.sort_bus_func)

        for user in users_want_public_transport:
            #get buses that pass by the user's house
            poss_buses = [bus for bus in bus_users if(
                user.house_node in bus.schedule)]
            #there are no buses that pass through his house
            if(len(poss_buses) == 0):
                users_not_matched.append(user)
                continue
            else:
                #there are buses that pass through his house
                min_time = 1000  # big number
                min_time_index = -1
                for i, bus in enumerate(poss_buses):
                    wait_time = abs(
                        bus.schedule[user.house_node] - user.start_time)
                    #if the bus is within the time window
                    if(wait_time <= (MAX_WAITING_TIME * user.personality.willingness_to_wait)):
                        #if this bus get to the user faster
                        if(wait_time < min_time):
                            #and the bus has space for the user
                            if(bus.available_seats > 0):
                                min_time_index = i
                                min_time = wait_time
                #if he was matched
                if(min_time_index != -1):
                    poss_buses[min_time_index].users_to_pick_up.append(user)
                    poss_buses[min_time_index].available_seats -= 1
                else:
                    users_not_matched.append(user)
        return users_not_matched

        # for driver in bus_drivers:
        #     # if(driver.available_seats == 0):
        #     #     continue
        #     #find users which the house node belongs to the bus route
        #     pickup = []
        #     possible_pickup = []
        #     for user in users_want_public_transport:
        #         # and in_time_slot(user, driver)
        #         if(user.house_node in driver.route and self.in_time_slot(user,driver)):
        #             possible_pickup.append(user)

        #     for user in possible_pickup:
        #         if(driver.available_seats > 0):
        #             pickup.append(user)
        #             driver.available_seats -= 1

        #     driver.users_to_pick_up = pickup
        #     users_want_public_transport = list(set(users_want_public_transport) - set(pickup))
        # return users_want_public_transport

    #Check if user can use the bicycle service
    def can_cycle(self, users: List['User']):
        allowed_users = []
        for user in users:
            has_path = self.graph.check_has_route(user.house_node, "bike")
            #if the user has at least one path to go from their house to the destination then it can use their bicycle
            if(has_path and user.has_bike):
                allowed_users.append(user)
        return allowed_users

    #Create buses actors
    def create_buses(self):
        print("im in create buses \n")
        peaks = get_traffic_peaks(self.traffic_distribution)
        bus_times = []
        value = 0.0
        peak_time = False

        #Get routes from input file
        existing_routes = self.input_config["buses"]

        while value < 16:
            bus_times.append(value)
            #check if the time is between any of peaks +- standard deviation
            for peak, std in peaks:
                if((peak - std) <= value < (peak + std)):
                    #If in peak time the bus comes every 10 minutes
                    value += 10/60
                    peak_time = True
            #If not in peak time the bus comes every 30 minutes
            if(peak_time == False):
                value += 0.5
            #reset the flag
            peak_time = False

        bus_users = []
        #for all the bus_times
        # for all bus routes
        # create a default user with the route and the time
        for time in bus_times:
            for route in existing_routes:
                us = User.default()
                #inser the right provider - Bus provider
                us.provider = STCP()
                #insert start time
                us.start_time = time
                #insert route str
                us.route_name = route
                #insert route list[int] which is a list of nodes
                us.route = existing_routes[route]
                #available seats
                us.available_seats = 50
                #capacity
                us.capacity = 50

                bus_users.append(us)
        return bus_users

    #Reset some variables between each new run day
    def reset(self):
        for bus_driver in self.bus_users:
            bus_driver.available_seats = bus_driver.capacity
            bus_driver.users_to_pick_up = []
        for user in self.users:
            user.available_seats = user.capacity
            user.users_to_pick_up = []
            user.mean_transportation = ""
            user.credits_won_round = 0

        self.parking_lot.available_seats_personal = self.parking_lot.lot_capacity - \
            self.parking_lot.lot_capacity_shared
        self.parking_lot.available_seats_shared = self.parking_lot.lot_capacity_shared

        for key in self.credits_used_run.keys():
            self.credits_used_run[key] = 0

    #Insert into each of the learner agent's training set, the utility for that action when the user was unmatched
    def unviable_choice_dict(self, dict_agents, users: List['User'], action_index: int):
        for user in users:
            agents = dict_agents[user.user_id]
            for agent in agents:
                agent.update_replay_memory((user.get_user_current_state(
                ), action_index, self.input_config["users"]["unviable_choice"], 1, True))

    #Insert into the learner agent's training set, the utility for that action when the user was unmatched

    def unviable_choice(self, agents: List[DQNAgent], users: List['User'], action_index: int):
        if(len(agents) == 1):
            agent = agents[0]
            for user in users:
                agent.update_replay_memory((user.get_user_current_state(
                ), action_index, self.input_config["users"]["unviable_choice"], 1, True))
        else:
            for user in users:
                agent = agents[user.user_id-1]
                agent.update_replay_memory((user.get_user_current_state(
                ), action_index, self.input_config["users"]["unviable_choice"], 1, True))

    #Create buses schedule
    def create_buses_schedule(self, buses: List[User]):
        #go through all the buses and create their schedule
        for bus in buses:
            schedule = {}
            for i, node in enumerate(bus.route):
                schedule[node] = bus.start_time + i/STCP().get_speed()
            bus.schedule = schedule

    #Check if user can use their bicycle
    def assess_can_cycle_var(self, user: User):
        return (user.has_bike and self.graph.check_has_route(user.house_node, "bike"))

    #Check if user can possibly walk to the destination
    def assess_can_walk_var(self, user: User):
        return (self.graph.check_has_route(user.house_node, "walk"))

    #Asses bicycling and walking variables values for all the users of the society
    def assess_cycle_walk(self, users: List[User]):
        for user in self.users:
            can_cycle = self.assess_can_cycle_var(user)
            can_walk = self.assess_can_walk_var(user)
            user.can_cycle = can_cycle
            user.can_walk = can_walk

    ###############################################################################
    ################### Learning techniques ########################################
    ###############################################################################

    #Run "agent cluster" technique
    #There are a total of 5 learning agents that apply deep q-learning for choosing the mobility service.
    #Each agent is in charge of learning and making decisions for a single cluster
    def run_agent_cluster(self, agents: List[DQNAgent]):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []
        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        agent_clusters_dict = {
            "cluster_0": 0,
            "cluster_1": 1,
            "cluster_2": 2,
            "cluster_3": 3,
            "cluster_4": 4
        }

        if(self.first_run):
            print(" first run")
            print(" am in run agent cluster")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()

            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            self.reset()
        self.choose_mode_agent_cluster(agents)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index

                for cluster in agent_clusters_dict.keys():
                    agent_index = agent_clusters_dict[cluster]
                    agent = agents[agent_index]
                    public_transport_c = [
                        user for user in public_transport_unmatched if(user.cluster == cluster)]
                    if(len(public_transport_c) > 0):
                        self.unviable_choice(
                            [agent], public_transport_c, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                for cluster in agent_clusters_dict.keys():
                    agent_index = agent_clusters_dict[cluster]
                    agent = agents[agent_index]
                    ride_sharing_c = [
                        user for user in ride_sharing_unmatched if(user.cluster == cluster)]
                    if(len(ride_sharing_c) > 0):
                        self.unviable_choice(
                            [agent], ride_sharing_c, ride_sharing_index)

        create_actor_events = self.create_actors_events(users_turn_actors)

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

        for ae in create_actor_events:
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

        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    actor.user.add_credits(actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #check if the discount is bigger than the cost of the trip.
                        # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(discount, driver_cost)

                        #this means the user has the minimum amount of credits and will have a discounted trip
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))

                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)

        #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i

            agent_index = agent_clusters_dict[user_info["user"].cluster]
            agent = agents[agent_index]

            agent.update_replay_memory(
                (current_state, provider_index,
                    user_info["utility"], 1, True))

        for agent in agents:
            agent.train(True, 1)

        self.runn += 1

        return final_users

    #Run "ensemble" technique
    #There are a total of 4 learning agents that apply deep q-learning for choosing the mobility service.
    #Each agent is in charge of learning and making decisions for all the users, knowing only a segment of the utility function
    def run_ensemble(self, agents: List[DQNAgent]):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []
        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        if(self.first_run):
            print(" first run")
            print("am in run ensemble")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()
            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            self.reset()

        self.choose_mode_ensemble(agents)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index
                for agent in agents:
                    self.unviable_choice(
                        [agent], public_transport_unmatched, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                for agent in agents:
                    self.unviable_choice(
                        [agent], ride_sharing_unmatched, ride_sharing_index)

        create_actor_events = self.create_actors_events(users_turn_actors)

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

        for ae in create_actor_events:
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

        #Sort actors using time they arrived to the destination
        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    #this means that the user doesn't have the minimum amount of credits to have discount
                    #since the user didnt spent credits, he will gain credits for his trip
                    actor.user.add_credits(actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                user_info["cost_util"] = actor.user.cost_util(commute_out)
                user_info["time_util"] = actor.user.time_util(commute_out)
                user_info["environmental_util"] = actor.user.social_util(
                    commute_out)
                user_info["comfort_util"] = actor.user.comfort_util(
                    commute_out)

                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #this means the user has the minimum amount of credits and will have a discounted trip
                        #check if the discount is bigger than the cost of the trip.
                        # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(
                            discount, driver_cost)
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)

                user_info["cost_util"] = actor.user.cost_util(commute_out)
                user_info["time_util"] = actor.user.time_util(commute_out)
                user_info["environmental_util"] = actor.user.social_util(
                    commute_out)
                user_info["comfort_util"] = actor.user.comfort_util(
                    commute_out)

                #update user credits
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)

                    user_info["cost_util"] = rider.cost_util(commute_out)
                    user_info["time_util"] = rider.time_util(commute_out)
                    user_info["environmental_util"] = rider.social_util(
                        commute_out)
                    user_info["comfort_util"] = rider.comfort_util(commute_out)

                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:

                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)

                    user_info["cost_util"] = rider.cost_util(commute_out)
                    user_info["time_util"] = rider.time_util(commute_out)
                    user_info["environmental_util"] = rider.social_util(
                        commute_out)
                    user_info["comfort_util"] = rider.comfort_util(commute_out)

                    final_users.append(user_info)

        #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        users_copy = self.users
        final_users_copy = final_users

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i

            for index, agent in enumerate(agents):
                if(index == AGENT_COST_INDEX):
                    util = user_info["cost_util"]
                elif(index == AGENT_TIME_INDEX):
                    util = user_info["time_util"]
                elif(index == AGENT_ENVIRONMENT_INDEX):
                    util = user_info["environmental_util"]
                elif(index == AGENT_COMFORT_INDEX):
                    util = user_info["comfort_util"]

                agent.update_replay_memory(
                    (current_state, provider_index,
                        util, 1, True))

        for agent in agents:
            agent.train(True, 1)

        self.runn += 1

        return final_users

    #Create agent population from file information

    def import_pop(self):
        print("I am importing a population \n")
        with open(self.save_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1

                personality = Personality(
                    float(row["willingness_to_pay"]), float(row["willingness_to_wait"]), float(
                        row["awareness"]), float(row["comfort_preference"]),
                    float(row["dim1"]), float(row["dim2"]), float(row["dim3"]), float(row["dim4"]), float(row["dim5"]), float(row["dim6"]))

                user = User(personality, float(row["time"]), row["chosen_cluster"], row["chosen_course"], row["chosen_grade"], float(row["salary"]),
                            float(row["budget"]), int(row["available_seats"]), int(row["distance_from_destination"]), eval(row["has_bike"]), eval(row["has_private"]))
                user.house_node = int(row["house_node"])
                user.can_cycle = eval(row["can_cycle"])
                user.can_walk = eval(row["can_walk"])
                user.num_friends = int(row["num_friends"])
                user.user_id = int(row["user_id"])
                user.friends_ids = row["friends_ids"]
                self.users.append(user)
                line_count += 1

        print(f'Processed {line_count} lines.')

        for user in self.users:
            user.friends_ids = user.friends_ids.replace('[', '')
            user.friends_ids = user.friends_ids.replace(']', '')
            if(user.friends_ids == ""):
                user.friends = []
            else:
                split = user.friends_ids.split(',')
                f_ids = [int(i) for i in split]
                user.friends = [f for f in self.users if(f.user_id in f_ids)]

    #Saving agent population information to a file

    def save_pop(self):
        print("I am saving the population \n")

        fields = ["willingness_to_pay", "willingness_to_wait", "awareness", "comfort_preference", "dim1", "dim2", "dim3", "dim4", "dim5","dim6",
                  "time", "chosen_cluster", "chosen_course", "chosen_grade", "salary", "budget", "available_seats",
                  "distance_from_destination", "has_bike", "has_private", "house_node", "can_cycle", "can_walk", "num_friends", "user_id", "friends_ids"]

        with open(self.save_file, "w+") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for user in self.users:
                f_ids = []
                for friend in user.friends:
                    f_ids.append(friend.user_id)

                writer.writerow({
                    'willingness_to_pay': user.personality.willingness_to_pay,
                    'willingness_to_wait': user.personality.willingness_to_wait,
                    'awareness': user.personality.awareness,
                    'comfort_preference': user.personality.comfort_preference,
                    'dim1': user.personality.dim1,
                    'dim2': user.personality.dim2,
                    'dim3': user.personality.dim3,
                    'dim4': user.personality.dim4,
                    'dim5': user.personality.dim5,
                    'dim6': user.personality.dim6,
                    'time': user.start_time,
                    'chosen_cluster': user.cluster,
                    'chosen_course': user.course,
                    'chosen_grade': user.grade,
                    'salary': user.salary,
                    'budget': user.budget,
                    'available_seats': user.available_seats,
                    'distance_from_destination': user.distance_from_destination,
                    'has_bike': user.has_bike,
                    'has_private': user.has_private,
                    'house_node': user.house_node,
                    'can_cycle': user.can_cycle,
                    'can_walk': user.can_walk,
                    'num_friends': user.num_friends,
                    'user_id': user.user_id,
                    'friends_ids': f_ids
                })

    #Run "default" technique
    #There is only one learner agent that applies deep q-learning to the population's mobility choice
    def run(self, agent: DQNAgent):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        if(self.first_run):
            print(" first run")
            print(" default run")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()

            if(self.save_population):
                self.save_pop()

            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            self.reset()
        self.choose_mode(agent)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        #Create list of users that will be "turned into" actor
        # Users that chose their private vehicle will be actors
        # Users which are the Drivers of their ride sharing group will also be actors
        # Users which are riders, will not be actors
        #Everyone else is not an actor

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index
                self.unviable_choice(
                    [agent], public_transport_unmatched, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                self.unviable_choice(
                    [agent], ride_sharing_unmatched, ride_sharing_index)

        create_actor_events = self.create_actors_events(users_turn_actors)

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

        for ae in create_actor_events:
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

        #Add parking cost according to arrival time
        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    actor.user.add_credits(actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #this means the user has the minimum amount of credits and will have a discounted trip
                        #check if the discount is bigger than the cost of the trip.
                        # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(
                            discount, driver_cost)
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)

        #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        users_copy = self.users
        final_users_copy = final_users

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i

            agent.update_replay_memory(
                (current_state, provider_index,
                    user_info["utility"], 1, True))
        agent.train(True, 1)

        self.runn += 1

        return final_users

    def run_every_agent_nn(self, agents: List[DQNAgent]):
            # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        if(self.first_run):
            print(" first run")
            print(" tou no run every agent nn")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()

            if(self.save_population):
                self.save_pop()

            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            self.reset()
        self.choose_mode_every_agent_nn(agents)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        #Create list of users that will be "turned into" actor
        # Users that chose their private vehicle will be actors
        # Users which are the Drivers of their ride sharing group will also be actors
        # Users which are riders, will not be actors
        # Users that weren't matched up are TBD (To Be Determined) what happens to them
        # Users that chose public transport are TBI (To Be Implemented)

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        # print("antes de ver se temos stcp ", len(users_turn_actors))

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index
                self.unviable_choice(
                    agents, public_transport_unmatched, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                self.unviable_choice(
                    agents, ride_sharing_unmatched, ride_sharing_index)

        # print("depois de ver se temos stcp ", len(users_turn_actors))

        create_actor_events = self.create_actors_events(users_turn_actors)

        # print("users {}".format(len(self.users)))

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

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

        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    actor.user.add_credits(actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                #update dos creditos do user
                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                # print("travel cost: {}".format(travel_cost))
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    # print("tenho desconto")
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        # print("i gained ", actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #this means the user has the minimum amount of credits and will have a discounted trip
                        # print(discount)
                        # print("gastei creditos!")
                        #check if the discount is bigger than the cost of the trip.
                        # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(
                            discount, driver_cost)
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                #update dos creditos do user
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)

        #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        users_copy = self.users
        final_users_copy = final_users

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i
            agent = agents[user_info["user"].user_id-1]
            agent.update_replay_memory(
                (current_state, provider_index,
                    user_info["utility"], 1, True))

        for agent in agents:
            agent.train(True, 1)

        self.runn += 1

        return final_users

    def run_every_agent_ensemble(self, dict_agents):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)
        # for edge in self.graph.graph.edges:
        #     print(edge)

        if(self.first_run):
            print(" first run")
            print("am in run every agent ensemble")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()
            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            # self.users = self.users[1:]
            self.reset()

        self.choose_mode_every_agent_ensemble(dict_agents)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index
                self.unviable_choice_dict(
                    dict_agents, public_transport_unmatched, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                self.unviable_choice_dict(
                    dict_agents, ride_sharing_unmatched, ride_sharing_index)

        create_actor_events = self.create_actors_events(users_turn_actors)

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

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

        #Sort actors using time they arrived to the destination
        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    #this means that the user doesn't have the minimum amount of credits to have discount
                    #since the user didnt spent credits, he will gain credits for his trip
                    actor.user.add_credits(actor.provider.get_credits())
                    # print("i gained ", actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                user_info["cost_util"] = actor.user.cost_util(commute_out)
                user_info["time_util"] = actor.user.time_util(commute_out)
                user_info["environmental_util"] = actor.user.social_util(
                    commute_out)
                user_info["comfort_util"] = actor.user.comfort_util(
                    commute_out)

                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                # print("travel cost: {}".format(travel_cost))
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                # print("travel cost portion: %s", travel_cost_portion)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    # print("tenho desconto")
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        # print("i gained ", actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #this means the user has the minimum amount of credits and will have a discounted trip
                        # print(discount)
                        # print("gastei creditos!")
                        #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(
                            discount, driver_cost)
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)

                user_info["cost_util"] = actor.user.cost_util(commute_out)
                user_info["time_util"] = actor.user.time_util(commute_out)
                user_info["environmental_util"] = actor.user.social_util(
                    commute_out)
                user_info["comfort_util"] = actor.user.comfort_util(
                    commute_out)

                #update dos creditos do user
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)

                    user_info["cost_util"] = rider.cost_util(commute_out)
                    user_info["time_util"] = rider.time_util(commute_out)
                    user_info["environmental_util"] = rider.social_util(
                        commute_out)
                    user_info["comfort_util"] = rider.comfort_util(commute_out)

                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:

                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)

                    user_info["cost_util"] = rider.cost_util(commute_out)
                    user_info["time_util"] = rider.time_util(commute_out)
                    user_info["environmental_util"] = rider.social_util(
                        commute_out)
                    user_info["comfort_util"] = rider.comfort_util(commute_out)

                    final_users.append(user_info)

              #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        users_copy = self.users
        final_users_copy = final_users

        # for user_info in final_users:
        #     print(user_info["user"])
        #     print("cost util ", user_info["cost_util"])
        #     print("time util ",user_info["time_util"])
        #     print("env util ",user_info["environmental_util"])
        #     print("comf util " ,user_info["comfort_util"])
        #     print("willingness pay ",user_info["user"].personality.willingness_to_pay)
        #     print("will wait ", user_info["user"].personality.willingness_to_wait)
        #     print("awareness ", user_info["user"].personality.awareness)
        #     print("comfort ",user_info["user"].personality.comfort_preference)

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i

            agents = dict_agents[user_info["user"].user_id-1]

            for index, agent in enumerate(agents):
                if(index == AGENT_COST_INDEX):
                    util = user_info["cost_util"]
                elif(index == AGENT_TIME_INDEX):
                    util = user_info["time_util"]
                elif(index == AGENT_ENVIRONMENT_INDEX):
                    util = user_info["environmental_util"]
                elif(index == AGENT_COMFORT_INDEX):
                    util = user_info["comfort_util"]

                agent.update_replay_memory(
                    (current_state, provider_index,
                        util, 1, True))

        for key in dict_agents.keys():
            for agent in dict_agents[key]:
                agent.train(True, 1)

        self.runn += 1

        return final_users

    def run_every_agent_fast(self, agents: List[DQNAgent]):
            # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        if(self.first_run):
            print(" first run")
            print(" tou no run every agent fast")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
                # for user in self.users:
                #     user.has_private = True
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()

            if(self.save_population):
                self.save_pop()

            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            self.reset()
        self.choose_mode_every_agent_fast(agents)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        #Create list of users that will be "turned into" actor
        # Users that chose their private vehicle will be actors
        # Users which are the Drivers of their ride sharing group will also be actors
        # Users which are riders, will not be actors
        # Users that weren't matched up are TBD (To Be Determined) what happens to them
        # Users that chose public transport are TBI (To Be Implemented)

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        # print("antes de ver se temos stcp ", len(users_turn_actors))

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index
                self.unviable_choice(
                    agents, public_transport_unmatched, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                self.unviable_choice(
                    agents, ride_sharing_unmatched, ride_sharing_index)

        # print("depois de ver se temos stcp ", len(users_turn_actors))

        create_actor_events = self.create_actors_events(users_turn_actors)

        # print("users {}".format(len(self.users)))

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

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

        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    actor.user.add_credits(actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                # print("cost ", actor.cost)
                # print("pa ", actor.get_parking_cost())

                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)

                #update dos creditos do user
                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                # print("travel cost: {}".format(travel_cost))
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    # print("tenho desconto")
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        # print("i gained ", actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #this means the user has the minimum amount of credits and will have a discounted trip
                        # print(discount)
                        # print("gastei creditos!")
                        #check if the discount is bigger than the cost of the trip.
                        # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(
                            discount, driver_cost)
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                #update dos creditos do user
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            discount = min(discount, rider_cost)
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)

        #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        users_copy = self.users
        final_users_copy = final_users

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i
            agent = agents[user_info["user"].user_id-1]
            agent.update_replay_memory(
                (current_state, provider_index,
                    user_info["utility"], 1, True))

        for agent in agents:
            agent.train(True, 1)

        self.runn += 1

        return final_users

    def run_every_agent_fast_ensemble(self, dict_agents):
           # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)
        # for edge in self.graph.graph.edges:
        #     print(edge)

        if(self.first_run):
            print(" first run")
            print("am in run every agent fast ensemble")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Create distance dictionary
                self.create_dist_dict()
                #Assign house nodes to each user according to graph structure
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()
            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)
        else:
            # self.users = self.users[1:]
            self.reset()

        self.choose_mode_every_agent_ensemble(dict_agents)

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)

        #Take care of the ride sharing option - matching
        users_ride_sharing = []
        for user in self.users:
            if (user.mean_transportation == "sharedCar"):
                users_ride_sharing.append(user)

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(
                users_ride_sharing)

        cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        walkers = [user for user in self.users if(
            user.mean_transportation == "walk")]

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        users_turn_actors = []
        for user in self.users:
            if(user.mean_transportation == "car"):
                users_turn_actors.append(user)
            elif(user.mean_transportation == "sharedCar"):
                if(len(user.users_to_pick_up) > 0):
                    users_turn_actors.append(user)

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)

            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + ride_sharing_unmatched

        #Training the users to learn that the mode choice they made was not good, they weren't matched
        for index in range(0, len(self.providers)):
            if(self.providers[index].service == "bus"):
                public_index = index
                self.unviable_choice_dict(
                    dict_agents, public_transport_unmatched, public_index)
            elif(self.providers[index].service == "sharedCar"):
                ride_sharing_index = index
                self.unviable_choice_dict(
                    dict_agents, ride_sharing_unmatched, ride_sharing_index)

        create_actor_events = self.create_actors_events(users_turn_actors)

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["total"] = len(ride_sharing_unmatched) + \
            len(public_transport_unmatched)

        self.users_lost[self.runn] = lost

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

        #Sort actors using time they arrived to the destination
        self.actors.sort(key=self.sort_actors_fun)
        for a in self.actors:
            if (a.service == "car" or a.service == "sharedCar"):
                parking_cost = self.parking_lot.park_vehicle(a.service)
                a.add_parking_cost(parking_cost)

        final_users = []

        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                if(CREDIT_INCENTIVE):
                    #this means that the user doesn't have the minimum amount of credits to have discount
                    #since the user didnt spent credits, he will gain credits for his trip
                    actor.user.add_credits(actor.provider.get_credits())
                    # print("i gained ", actor.provider.get_credits())
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        actor.cost + actor.get_parking_cost(), actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                user_info["cost_util"] = actor.user.cost_util(commute_out)
                user_info["time_util"] = actor.user.time_util(commute_out)
                user_info["environmental_util"] = actor.user.social_util(
                    commute_out)
                user_info["comfort_util"] = actor.user.comfort_util(
                    commute_out)

                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.sharing_travel_cost() + actor.get_parking_cost()
                # print("travel cost: {}".format(travel_cost))
                travel_cost_portion = travel_cost / \
                    (len(actor.user.users_to_pick_up) + 1)
                # print("travel cost portion: %s", travel_cost_portion)
                driver_cost = travel_cost_portion - \
                    actor.calculate_transporte_subsidy(
                        len(actor.traveled_nodes)-1)
                if(CREDIT_INCENTIVE):
                    # print("tenho desconto")
                    discount = actor.user.credits_discount(
                        N_CREDITS_DISCOUNT, CREDIT_VALUE)
                    if(discount == -1):
                        #this means that the user doesn't have the minimum amount of credits to have discount
                        #since the user didnt spent credits, he will gain credits for his trip
                        actor.user.add_credits(actor.provider.get_credits())
                        # print("i gained ", actor.provider.get_credits())
                        commute_out = CommuteOutput(
                            driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        #this means the user has the minimum amount of credits and will have a discounted trip
                        # print(discount)
                        # print("gastei creditos!")
                        #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                        discount = min(
                            discount, driver_cost)
                        self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                        self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                        commute_out = CommuteOutput(
                            driver_cost - discount, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                else:
                    commute_out = CommuteOutput(
                        driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)

                user_info["cost_util"] = actor.user.cost_util(commute_out)
                user_info["time_util"] = actor.user.time_util(commute_out)
                user_info["environmental_util"] = actor.user.social_util(
                    commute_out)
                user_info["comfort_util"] = actor.user.comfort_util(
                    commute_out)

                #update dos creditos do user
                final_users.append(user_info)

                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    #calculate the time the user spent waiting (difference between user starting time and time when the actor arrived to the house node)
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    rider_cost = travel_cost_portion - \
                        actor.calculate_transporte_subsidy(
                            actor.rider_traveled_dist(rider.house_node))

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(rider_cost - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:
                        commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)

                    user_info["cost_util"] = rider.cost_util(commute_out)
                    user_info["time_util"] = rider.time_util(commute_out)
                    user_info["environmental_util"] = rider.social_util(
                        commute_out)
                    user_info["comfort_util"] = rider.comfort_util(commute_out)

                    final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)

                    if(CREDIT_INCENTIVE):
                        # print("tenho desconto")
                        discount = rider.credits_discount(
                            N_CREDITS_DISCOUNT, CREDIT_VALUE)
                        if(discount == -1):
                            #this means that the user doesn't have the minimum amount of credits to have discount
                            #since the user didnt spent credits, he will gain credits for his trip
                            rider.add_credits(actor.provider.get_credits())
                            # print("i gained ", actor.provider.get_credits())
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                        else:
                            #this means the user has the minimum amount of credits and will have a discounted trip
                            # print(discount)
                            # print("gastei creditos!")
                            #check if the discount is bigger than the cost of the trip.
                            # if discount is bigger than discount is now equal to the cost, to avoid getting "negative cost"
                            discount = min(
                                discount, actor.rider_cost(rider.house_node))
                            self.credits_used[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run[actor.service] += N_CREDITS_DISCOUNT
                            self.credits_used_run["total"] += N_CREDITS_DISCOUNT
                            commute_out = CommuteOutput(actor.rider_cost(rider.house_node) - discount, actor.rider_travel_time(
                                rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    else:

                        commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                            rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)

                    user_info["cost_util"] = rider.cost_util(commute_out)
                    user_info["time_util"] = rider.time_util(commute_out)
                    user_info["environmental_util"] = rider.social_util(
                        commute_out)
                    user_info["comfort_util"] = rider.comfort_util(commute_out)

                    final_users.append(user_info)

              #add social credits to everybody if SOCIAL_CREDITS is true
        if(CREDIT_INCENTIVE and SOCIAL_CREDIT):
            self.social_credits_computation()

        users_copy = self.users
        final_users_copy = final_users

        for user_info in final_users:
            current_state = user_info["user"].get_user_current_state()
            provider_index = -1
            for i in range(len(self.providers)):
                if type(self.providers[i]) is type(user_info["user"].provider):
                    provider_index = i

            agents = dict_agents[user_info["user"].user_id]

            for index, agent in enumerate(agents):
                if(index == AGENT_COST_INDEX):
                    util = user_info["cost_util"]
                elif(index == AGENT_TIME_INDEX):
                    util = user_info["time_util"]
                elif(index == AGENT_ENVIRONMENT_INDEX):
                    util = user_info["environmental_util"]
                elif(index == AGENT_COMFORT_INDEX):
                    util = user_info["comfort_util"]

                agent.update_replay_memory(
                    (current_state, provider_index,
                        util, 1, True))

        for key in dict_agents.keys():
            for agent in dict_agents[key]:
                agent.train(True, 1)

        self.runn += 1

        return final_users

    def run_descriptive(self):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []
        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        if(self.first_run):
            print("first run")
            print("tou no run descriptive")

            if(self.import_population):
                #Creates users using an existent file
                self.import_pop()
                #Create distance dictionary
                self.create_dist_dict()
            else:
                self.users = self.create_users()
                #Assign house nodes to each user according to graph structure
                self.create_dist_dict()
                self.add_house_nodes()
                self.assess_cycle_walk(self.users)
                self.create_friends()
            self.bus_users = self.create_buses()
            self.create_buses_schedule(self.bus_users)

            if(self.save_population):
                self.save_pop()
        else:
            # self.users = self.users[1:]
            self.reset()

        self.choose_mode_descriptive()

        #Take care of the public transport option - match users to buses
        users_public_transport = []
        for user in self.users:
            if (user.mean_transportation == "bus"):
                users_public_transport.append(user)
         # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        possible_cyclists = [user for user in self.users if(
            user.mean_transportation == "bike")]

        #check if people that wish to cycle exist, if so check if its possible, if they have a path
        cyclists_not_possible = []
        cyclists = []
        if(len(possible_cyclists) > 0):
            cyclists = self.can_cycle(possible_cyclists)
            cyclists_not_possible = list(
                set(possible_cyclists) - set(cyclists))

        walkers = []
        possible_walkers = [user for user in self.users if (
            user.mean_transportation == "walk")]
        walking_not_possible = [user for user in self.users if(
            user.mean_transportation == "walk" and not user.can_walk)]
        if(len(possible_walkers) > 0):
            walkers = list(set(possible_walkers) - set(walking_not_possible))

        # Create the Simulation Actors
        event_queue = PriorityQueue()

        #Create list of users that will be "turned into" actor
        # Users that chose their private vehicle will be actors
        # Users which are the Drivers of their ride sharing group will also be actors
        # Users which are riders, will not be actors
        # Users that weren't matched up are TBD (To Be Determined) what happens to them
        # Users that chose public transport are TBI (To Be Implemented)

        users_turn_actors = []
        users_no_car_but_chose_car = []

        #Append actors to represent Personal Vehicle (Car)
        for user in self.users:
            if(user.mean_transportation == "car"):
                if(user.has_private):
                    users_turn_actors.append(user)
                else:
                    users_no_car_but_chose_car.append(user)

        services = []
        for provider in self.providers:
            services.append(provider.service)

        public_transport_unmatched = []
        for service in services:
            if(service == "bus"):
                # add bus drivers to the users who will become actors
                public_transport_unmatched = self.public_transport_matching(
                    users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)
            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists
            if(service == "walk"):
                users_turn_actors = users_turn_actors + walkers

        # print("depois de ver se temos stcp ", len(users_turn_actors))

        create_actor_events = self.create_actors_events(users_turn_actors)

        # print("users {}".format(len(self.users)))

        lost = dict()
        lost["car"] = len(users_no_car_but_chose_car)
        lost["bus"] = len(public_transport_unmatched)
        lost["bike"] = len(cyclists_not_possible)
        lost["walk"] = len(walking_not_possible)
        lost["total"] = len(users_no_car_but_chose_car) + len(public_transport_unmatched) + \
            len(cyclists_not_possible) + len(walking_not_possible)

        self.users_lost[self.runn] = lost

        # print("users lost")
        # print(self.users_lost)
        self.runn += 1
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
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                commute_out = CommuteOutput(
                    actor.cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                # user_info["fastest_time"] = self.fastest_private_travel_time(actor.user.house_node)
                #update dos creditos do user
                final_users.append(user_info)
            elif(actor.service == "bus"):
                #go through each user this actor represents
                for rider in actor.user.users_to_pick_up:
                    time_waiting = actor.driver_reached_pick_up(
                        rider.house_node) - rider.start_time
                    rider.time_spent_waiting = max(time_waiting, 0)
                    commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                        rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    # user_info["fastest_time"] = self.fastest_private_travel_time(
                    #     rider.house_node)
                    final_users.append(user_info)

        return final_users

    def create_actors_events(self, users: [User]) -> List[CreateActorEvent]:
        """Returns all scheduled CreateActorEvents"""
        return [
            CreateActorEvent(
                user.start_time, self.actor_constructor, user)
            for user in users
        ]

    # def create_accident_events(self) -> List[AccidentEvent]:
    #     return [
    #         # AccidentEvent(10.0, (3, 6), 0.2)
    #     ]

    def create_users(self):
        users = []

        for _ in range(self.input_config["users"]["num_users"]):

            time = get_time_from_traffic_distribution(
                self.traffic_distribution)

            #personality params we had previously defined, namely, willingness_to_wait/pay, comfort preference
            personality_params = self.input_config["users"]["personality_params"]

            #####################################################################################################
            ###   WORK IN PROGRESS   ###########################

            #Salary distribution
            salary_params = self.input_config["users"]["salary_distribution"]

            dist = getattr(scipy.stats, salary_params["distrib"])

            shape_list = list(
                salary_params["shape"].values())

            # random_num = dist.rvs(a=*shape_list, scale=salary_params["scale_param"])

            salary = dist.rvs(
                *shape_list, scale=salary_params["scale_param"])
            min_salary = salary_params["min_salary"]
            max_salary = salary_params["max_salary"]

            if(salary < min_salary):
                salary = min_salary
            else:
                if(salary > max_salary):
                    salary = max_salary

            budget = salary * (salary_params["budget_percent"]/100)

            #Willingness to pay is derived from the Budget
            willingness_to_pay = (salary - min_salary) / \
                (max_salary - min_salary)

            willingness_to_wait = np.random.normal(
                personality_params["willingness_to_wait"]["mean"],
                personality_params["willingness_to_wait"]["stand_div"]
            )

            awareness = np.random.normal(
                personality_params["awareness"]["mean"],
                personality_params["awareness"]["stand_div"]
            )

            comfort_preference = np.random.normal(
                personality_params["comfort_preference"]["mean"],
                personality_params["comfort_preference"]["stand_div"]
            )

            willingness_to_pay = min(max([0.01, willingness_to_pay]), 1.0)
            willingness_to_wait = min(max([0.01, willingness_to_wait]), 1.0)
            comfort_preference = min(max([0.01, comfort_preference]), 1.0)
            awareness = min(max([0.000, awareness]), 1.0)

            ###################################################################################################################

            clusters_distribs = self.input_config["users"]["clusters_distribution"]

            #Allocate users to clusters
            random_value_cluster = np.random.uniform()
            chosen_cluster = ""

            for (cluster, cluster_d) in clusters_distribs.items():
                if(random_value_cluster <= cluster_d):
                    chosen_cluster = cluster
                    break

            #Choose if the user has a private vehicle or not according to cluster ratio
            has_private = 1 if np.random.uniform(
            ) < self.input_config["users"]["clusters"][chosen_cluster]["has_private"]["ratio"] else 0

            #Choose if the user has a bicycle
            has_bike = 1 if np.random.uniform(
            ) < self.input_config["users"]["has_bike"] else 0

            #Assign the available seats in the private vehicle for each user who has a private vehicle according to the cluster
            if(has_private):
                seats_num = list(
                    (self.input_config["users"]["clusters"][chosen_cluster]["seat_probs"]).keys())
                seats_percentages = (
                    self.input_config["users"]["clusters"][chosen_cluster]["seat_probs"]).values()

                available_seats = random.choices(
                    seats_num, weights=seats_percentages, k=1)
                for s in available_seats:
                    available_seats = int(s)
            else:
                available_seats = 0

            #Get distance from the users' home to the destination, according to cluster distribution
            distance_from_destination_info = self.input_config[
                "users"]["clusters"][chosen_cluster]["distance"]

            dist = getattr(
                scipy.stats, distance_from_destination_info["distrib"])
            #Get shape params
            shape_list = list(
                distance_from_destination_info["shape"].values())

            random_num = dist.rvs(
                *shape_list, loc=distance_from_destination_info["mean"], scale=distance_from_destination_info["stand_div"])

            random_num = int(round(random_num))
            if (random_num < MIN_HOUSE_DISTANCE):
                random_num = MIN_HOUSE_DISTANCE
            elif(random_num > MAX_HOUSE_DISTANCE):
                random_num = MAX_HOUSE_DISTANCE

            distance_from_destination = random_num

            #Allocate users to grades and courses
            courses_distribs = self.input_config["users"]["courses_distribution"]

            random_value_course = np.random.uniform()
            chosen_course = ""

            for (course, course_d) in courses_distribs.items():
                if(random_value_course <= course_d):
                    chosen_course = course
                    break

            grades_distribs = self.input_config["users"]["courses"][chosen_course]

            random_value_grade = np.random.uniform()
            chosen_grade = ""

            for (grade, grade_d) in grades_distribs.items():
                if(random_value_grade <= grade_d):
                    chosen_grade = grade
                    break

            #get random values for the factors according to the distributions (change these to the beggining so that it's clearer which factors exist)

            chosen_cluster_info = self.input_config["users"]["clusters"][chosen_cluster]
            user_factors = ["dim1", "dim2",
                            "dim3", "dim4", "dim5","dim6"]
            user_factors_values = {
                "dim1": DEFAULT_VALUE_NUM,
                "dim2": DEFAULT_VALUE_NUM,
                "dim3": DEFAULT_VALUE_NUM,
                "dim4": DEFAULT_VALUE_NUM,
                "dim5": DEFAULT_VALUE_NUM,
                "dim6": DEFAULT_VALUE_NUM
            }

            for factor in user_factors:

                dist = getattr(
                    scipy.stats, chosen_cluster_info[factor]["distrib"])
                #Get shape params
                shape_list = list(
                    chosen_cluster_info[factor]["shape"].values())

                random_num = dist.rvs(
                    *shape_list, loc=chosen_cluster_info[factor]["mean"], scale=chosen_cluster_info[factor]["stand_div"])

                user_factors_values[factor] = random_num
                #Get distribution information for that factor

            #
            # Check if values are within the limits
            #
            # Add factors to the personality of the users(?)
            # add cluster info to the users(?)

            personality = Personality(willingness_to_pay, willingness_to_wait, awareness, comfort_preference,
                                      user_factors_values["dim1"], user_factors_values["dim2"], user_factors_values["dim3"], user_factors_values["dim4"], user_factors_values["dim5"], user_factors_values["dim6"])
            user = User(personality, time, chosen_cluster,
                        chosen_course, chosen_grade, salary, budget, available_seats, distance_from_destination, bool(has_bike), bool(has_private))

            # se estiverem entao proximo passo é adicionar tambem informaçao de ano e curso!
            users.append(user)

        return users

    def create_dist_dict(self):
        #initialize dictionary
        self.distance_dict["1-3"] = 0
        self.distance_dict["3-5"] = 0
        self.distance_dict["5-10"] = 0
        self.distance_dict["10-20"] = 0
        self.distance_dict["20-30"] = 0
        self.distance_dict["30"] = 0

        for user in self.users:
            distance = user.distance_from_destination
            if(distance >= 30):
                self.distance_dict["30"] += 1
            elif(distance >= 20 and distance < 30):
                self.distance_dict["20-30"] += 1
            elif(distance >= 10 and distance < 20):
                self.distance_dict["10-20"] += 1
            elif(distance >= 5 and distance < 10):
                self.distance_dict["5-10"] += 1
            elif(distance >= 3 and distance < 5):
                self.distance_dict["3-5"] += 1
            elif(distance >= 1 and distance < 3):
                self.distance_dict["1-3"] += 1

    def fastest_private_travel_time(self, house_node: int):
        dist = nx.shortest_path_length(
            self.graph.graph, source=house_node, target=self.graph.nend)
        car_p = [provider for provider in self.providers if provider.service == "car"]
        speed = car_p[0].get_speed()
        time = dist/speed
        return time

    def social_credits_computation(self):
        #add social credits to everybody if SOCIAL_CREDITS is true
        for actor in self.actors:
            if(actor.service == "car" or actor.service == "bike" or actor.service == "walk"):
                continue
            elif(actor.service == "bus"):
                for rider in actor.user.users_to_pick_up:
                    soc_cre = rider.get_social_credits(SOCIAL_PERCENT)
                    rider.add_social_credits(soc_cre)
                    # print(rider)
                    # print("i got {} social credits".format(soc_cre))
            elif(actor.service == "sharedCar"):
                soc_cre = actor.user.get_social_credits(SOCIAL_PERCENT)
                actor.user.add_social_credits(soc_cre)
                # print(actor.user)
                # print("i got {} social credits".format(soc_cre))
                for rider in actor.user.users_to_pick_up:
                    soc_cre = rider.get_social_credits(SOCIAL_PERCENT)
                    rider.add_social_credits(soc_cre)
                    # print(rider)
                    # print("i got {} social credits".format(soc_cre))
            # print("\n")
