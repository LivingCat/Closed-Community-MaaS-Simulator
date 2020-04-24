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
from provider import Provider, STCP, Personal, Friends
from DeepRL import DQNAgent
from actor import Actor
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy

DEFAULT_VALUE_NUM = -100.0
DEFAULT_VALUE_STRING = ""


class Simulator:
    """Runs a simulation from a given set of parameters"""

    def __init__(self,
                 config,
                 input_config,
                 actor_constructor,
                 providers,
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
        self.providers = providers

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


    def create_friends(self):
        print("im create friends")
        friend_distrib_info = self.input_config["users"]["friends_distribution"]

        dist = getattr(scipy.stats, friend_distrib_info["distrib"])

        #Go through all the users of the system and input num_friends for each of them following the distribution
        for user in self.users:
            num_friend = int(dist.rvs(loc=friend_distrib_info["mean"], scale=friend_distrib_info["stand_div"]))
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
            friend_categories = [0,1,2,3]
            friend_categories_probs = [0.5,0.2,0.2,0.1]

            cats = random.choices(friend_categories,weights = friend_categories_probs, k = user.num_friends)
            # print("user")
            # print(user.course)
            # print(user.grade)
            # print(cats)
            for cat in cats:
                if(cat == 0):
                    for poss_friend in self.users:
                       if(poss_friend != user and user.course == poss_friend.course and user.grade == poss_friend.grade and poss_friend.num_friends > 0 and (poss_friend not in user.friends) ):
                           user.friends.append(poss_friend)
                           user.num_friends -= 1
                           poss_friend.friends.append(user)
                           poss_friend.num_friends -= 1
                           break
                elif(cat == 1):
                    for poss_friend in self.users:
                        grade_num = int(user.grade[(user.grade.index("_") + 1)])
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

        # for user in self.users:
        #     print("user")
        #     print(user.friends)
        #     print(user.num_friends)

        return True

    def run(self, agent: DQNAgent):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []
        self.users = self.create_users(agent)
        self.create_friends()

        for user in self.users:
            user.pprint()
        exit()
        #############################################################################################################
        #######################################################################################

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
            commute_out = CommuteOutput(
                actor.cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
            user_info = dict()
            user_info["user"] = actor.user
            user_info["commute_output"] = commute_out
            user_info["utility"] = actor.user.calculate_utility_value(
                commute_out)
            final_users.append(user_info)

            # print("mean: {}  utility: {} ".format(commute_out.mean_transportation, user_info["utility"]))

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

        # self.draw_graph()
        return final_users

    def create_actors_events(self, users: [User]) -> List[CreateActorEvent]:
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

            time = get_time_from_traffic_distribution(
                self.traffic_distribution)

            #personality params we had previously defined, namely, willingness_to_wait/pay, comfort preference and also has_private
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

            willingness_to_pay = min(max([0.0001, willingness_to_pay]), 1)
            willingness_to_wait = min(max([0.0001, willingness_to_wait]), 1)
            comfort_preference = min(max([0.0001, comfort_preference]), 1)
            awareness = min(max([0.000, awareness]), 1)

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

            #Assign the available seats in the private vehicle for each user who has a private vehicle according to the cluster
            if(has_private):
                seats_num = list((self.input_config["users"]["clusters"][chosen_cluster]["seat_probs"]).keys())
                seats_percentages = (self.input_config["users"]["clusters"][chosen_cluster]["seat_probs"]).values()
                # print(seats_num)
                # print(seats_percentages)

                available_seats = random.choices(
                    seats_num, weights=seats_percentages, k=1)
            else:
                available_seats = 0


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
            user_factors = ["friendliness", "suscetible",
                            "transport", "willing", "urban"]
            user_factors_values = {
                "friendliness": DEFAULT_VALUE_NUM,
                "suscetible": DEFAULT_VALUE_NUM,
                "transport": DEFAULT_VALUE_NUM,
                "willing": DEFAULT_VALUE_NUM,
                "urban": DEFAULT_VALUE_NUM
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

            personality = Personality(willingness_to_pay, willingness_to_wait, awareness, comfort_preference, bool(has_private),
                                      user_factors_values["friendliness"], user_factors_values["suscetible"], user_factors_values["transport"], user_factors_values["urban"], user_factors_values["willing"])
            user = User(personality, time, chosen_cluster,
                        chosen_course, chosen_grade, salary, budget, available_seats)

            # se estiverem entao proximo passo é adicionar tambem informaçao de ano e curso!

            #Users chooses action to take (ask Tiago why this is here and where is the learning part)
            while True:
                if random.random() > agent.epsilon:  # dizia np.random.random
                    # Get action from Q table
                    current_state = np.array(user.get_user_current_state())
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, agent.output_dim)

                provider = self.providers[action]
                if((not user.personality.has_private) and provider.name == "Personal"):
                    agent.update_replay_memory(
                        (user.get_user_current_state(), action,
                         self.input_config["users"]["punishment_doesnt_have_car"], 1, True))
                    agent.train(True, 1)
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider
            users.append(user)

        return users
