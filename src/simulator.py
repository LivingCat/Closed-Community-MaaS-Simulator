"""
Simulation process.
From micro-level decision making and learning, to macro-level simulation of Users on a graph network.
"""
from typing import List
from queue import PriorityQueue
from event import CreateActorEvent
from graph import RoadGraph
from utils import MultimodalDistribution, get_time_from_traffic_distribution, get_traffic_peaks
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
MIN_HOUSE_DISTANCE = 1
MAX_HOUSE_DISTANCE = 30
MAX_WAITING_TIME = 0.10


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
        self.first_run = False
        self.runn = 0
        self.users_lost = dict(dict())

        random.seed(seed)
        # plt.ion()

    def draw_graph(self):

        # #Trying to display graph
        # # plt.ion()
        # display_graph = self.graph.graph

        # colors = [display_graph[u][v]['color'] for u, v in display_graph.edges]

        # # weights = [display_graph[u][v]['weight']
        # #            for u, v in display_graph.edges]

        # print(display_graph.number_of_edges())

        # pos = nx.spring_layout(display_graph)
        # elarge = display_graph.edges(data=True)

        # # nodes
        # nx.draw_networkx_nodes(display_graph, pos, node_size=700)

        # # edges
        # nx.draw_networkx_edges(display_graph, pos, edgelist=elarge,
        #                        width=6, edge_color=colors)

        # # labels
        # nx.draw_networkx_labels(display_graph, pos, edgelist=elarge,
        #                         width=6)
        # print("vou desenhar o graph espero eu")
        # plt.axis('off')

        nx.draw(self.graph.graph, pos=nx.planar_layout(self.graph.graph))

        # display_graph.clear()
        plt.title("graph!!!")
        plt.draw()
        # plt.ioff()
        # plt.pause(0.01)
        plt.waitforbuttonpress(0)
        plt.clf()

    def calculate_distance_nodes_destination(self):
        distance = []
        for node in range(0, int(self.input_config["graph"]["num_nodes"])):
            dist = nx.shortest_path_length(self.graph.graph,source = node, target=self.graph.nend)
            distance.append(dist)
        return distance

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

        return True

    def choose_mode_descriptive(self):
        for user in self.users:
            percents_cluster = self.input_config["users"]["clusters"][user.cluster]["original_choices"]
            # provider = self.providers[action]
            # user.mean_transportation = provider.service
            # user.provider = provider
            
            for key in percents_cluster.keys():
                # print(key)
                key_copy = str(key)
                limits = key_copy.split("-")
                limits = [int(limit) for limit in limits]
                # print(percents_cluster[key])
                keys = percents_cluster[key].keys()
                values = percents_cluster[key].values()

                if(len(limits) == 2):
                    if(user.distance_from_destination >= limits[0] and user.distance_from_destination < limits[1]):
                        mode = random.choices(list(keys), list(values), k=1)
                        for provider in self.providers:
                            if(provider.service == mode[0]):
                                user.mean_transportation = provider.service
                                user.provider = provider
                        break
                #upper bound
                if(len(limits) == 1):
                    if(user.distance_from_destination >= limits[0]):
                        mode = random.choices(list(keys),list(values),k=1)
                        for provider in self.providers:
                            if(provider.service == mode[0]):
                                user.mean_transportation = provider.service
                                user.provider = provider
                        break
                    else: 
                        print("erro")
                
              

    def choose_mode(self, agent: DQNAgent):
        for user in self.users:
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
                #if the user chose to use "personal", as in, personal engine powered vehicle bu it doesnt own one, it gets a punishment
                if((not user.personality.has_private) and provider.name == "Personal"):
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
                else:
                    break
            user.mean_transportation = provider.service
            user.provider = provider

    def add_house_nodes(self):
        distances = self.calculate_distance_nodes_destination()
        for user in self.users:
            res_list = [i for i, value in enumerate(distances) if value == user.distance_from_destination]
            node = random.choice(res_list)
            user.add_house_node(node)   

    def sort_drivers_fun(self,driver: User):
        return driver.available_seats

    def sort_bus_func(self,bus_driver: User):
        return len(bus_driver.route)

    def ride_sharing_matching(self,users: List['User']):
        users_want_ride_sharing = users
        possible_drivers = [user for user in users if user.personality.has_private]
        possible_drivers.sort(reverse= True, key=self.sort_drivers_fun)

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
                # and (rider.start_time - driver.start_time) <= MAX_WAITING_TIME
                if(rider.house_node in all_nodes and rider != driver):
                    possible_pickup.append(rider)
            #pick random rider
            # while available seats or possible_pickup != vazio

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

            while (driver.available_seats > 0 and len(possible_pickup) > 0 ):
                #moves to the next node
                #checks the users it can pickup from the new node

                #start a new list for the possible riders
                possible_pickup = []
                new_node = pickup[-1].house_node
                poss_routes = self.graph.get_all_routes(new_node, pickup[-1].mean_transportation)
                all_nodes = [node for route in poss_routes for node in route]

                for rider in users_want_ride_sharing:
                    # and (rider.start_time - driver.start_time) <= MAX_WAITING_TIME
                    if(rider.house_node in all_nodes and rider != driver):
                        possible_pickup.append(rider)
                #pick random rider
                # while available seats or possible_pickup != vazio

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

    def in_time_slot(self,rider:User, driver: User):
        wait_time = MAX_WAITING_TIME * rider.willingness_to_wait
        return (abs(driver.start_time - rider.start_time) <= wait_time)


    def public_transport_matching(self, users: List['User'], bus_users: List['User']):
        users_want_public_transport = users
        bus_drivers = bus_users
        
        bus_drivers.sort(key=self.sort_bus_func)
        for driver in bus_drivers:
            # if(driver.available_seats == 0):
            #     continue
            #find users which the house node belongs to the bus route
            pickup = []
            possible_pickup = []
            for user in users_want_public_transport:
                # and in_time_slot(user, driver)
                if(user.house_node in driver.route): 
                    possible_pickup.append(user)

            for user in possible_pickup:
                if(driver.available_seats > 0):
                    pickup.append(user)
                    driver.available_seats -= 1
                    
            driver.users_to_pick_up = pickup
            users_want_public_transport = list(set(users_want_public_transport) - set(pickup))
        return users_want_public_transport

    def can_cycle(self,users: List['User']):
        allowed_users = []
        for user in users:
            has_path = self.graph.check_has_route(user.house_node, "bike")
            #if the user has at least one path to go from their house to the destination then it can use their bicycle
            if(has_path and user.has_bike):
                allowed_users.append(user)
        return allowed_users




    #depois ver porque se
    # users nao tem ninguem para pickup entao nao podem ir de ride sharing
    # se users nao fazem parte dos to-pick-up de outros users entao tb nao podem ir de ride sharing
    # se esses users tiverem carro podem ir sozinhos mas se nao tiverem idk

    #mudar como é que os atores fazem a rota

        # for user in self.users:
        #     print("its me user : {} \n", user)
        #     print("escolhi : {} \n ", user.mean_transportation)
        #     print("tenho carro? : {} \n", user.personality.has_private)
        #     print("vivo aqui: {} \n", user.house_node)
        #     print("vou buscar: {} \n", user.users_to_pick_up)

    def create_buses(self):
        print("im in create buses \n")
        peaks = get_traffic_peaks(self.traffic_distribution)
        bus_times = []
        value = 0.0
        peak_time = False

        #Get routes from input file
        existing_routes = self.input_config["buses"]

        #24
        while value < 0.5:
            bus_times.append(value)

            #check if the time is between any of peaks +- standard deviation
            for peak,std in peaks:
                if( (peak - std) <= value < (peak + std) ):
                    #If in peak time the bus comes every 15 minutes
                    value += 0.25
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

    def reset(self):
        for bus_driver in self.bus_users:
            bus_driver.available_seats = bus_driver.capacity
            bus_driver.users_to_pick_up = []
        for user in self.users:
            user.available_seats = user.capacity
            user.users_to_pick_up = []
        
    def run_descriptive(self):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)

        if(self.first_run):
            print("first run")
            self.users = self.create_users()
            #Assign house nodes to each user according to graph structure
            self.add_house_nodes()
            self.create_friends()
            self.bus_users = self.create_buses()
        else:
            # self.users = self.users[1:]
            self.reset()
        self.choose_mode_descriptive()
        for user in self.users:
            print(user)

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
                if(user.personality.has_private):
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

        # print("depois de ver se temos stcp ", len(users_turn_actors))

        create_actor_events = self.create_actors_events(users_turn_actors)

        # print("users {}".format(len(self.users)))

        lost = dict()
        lost["car"] = len(users_no_car_but_chose_car)
        lost["bus"] = len(public_transport_unmatched)
        lost["bike"] = len(cyclists_not_possible)

        self.users_lost[self.runn] = lost

        print("users lost")
        print(self.users_lost)

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
            #actor.cost e actor.spent_credits
            #TODO
            #diferenciar entre transporte privado e os outros
            #transporte privado é como está agora
            #transport coletivo:
            #for pelos user_to_pick_up
            # descobrir quando é que o actor chegou ao house_node do user (atraves dos actor.travelled_nodes), pegar no tempo, ver a diferença entre o tempo total e as horas a que ele chegou ao house node
            # mudar assim o actor. travel time e o actor.cost (actor.provider.get_cost(real time))
            # nao esquecer tb do subsidio
            # Ride Sharing:
            # igual ao transport coletivo

            if(actor.service == "car" or actor.service == "bike"):
                commute_out = CommuteOutput(
                    actor.cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
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
                    final_users.append(user_info)

        return final_users



    def run(self, agent: DQNAgent):
        # Empty actors list, in case of consecutive calls to this method
        self.actors = []

        # Cleaning road graph
        self.graph = RoadGraph(self.input_config)
        # for edge in self.graph.graph.edges:
        #     print(edge)

        # exit()

        if(self.first_run):
            print("first run")
            self.users = self.create_users()
            #Assign house nodes to each user according to graph structure
            self.add_house_nodes()
            self.create_friends()
            self.bus_users = self.create_buses()
        else:
            # self.users = self.users[1:]
            self.reset()
        self.choose_mode(agent)

        # with open("buses_info.txt", 'a+') as f:
        #     print("run ", self.runn, file=f)
        #     for bus in self.bus_users:
        #         print("available : \n", bus.available_seats, file=f)
        #         print("capacity : \n", bus.capacity, file=f)


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

        #############################################################################################################
        #######################################################################################

        # Create the Statistics module
        self.stats = self.stats_constructor(self.graph)

        # for user in users_ride_sharing:
        #     print("its me user : \n", user)
        #     print("escolhi : \n ", user.mean_transportation)
        #     print("tenho carro? : \n", user.personality.has_private)
        #     print("vivo aqui: \n", user.house_node)
        #     print("available seats: \n", user.available_seats)

        #check if people that wish to ride share exist, if so match them
        ride_sharing_unmatched = []
        if(len(users_ride_sharing) > 0):
            ride_sharing_unmatched = self.ride_sharing_matching(users_ride_sharing)

        possible_cyclists = [user for user in self.users if(user.mean_transportation == "bike")]

        #check if people that wish to cycle exist, if so check if its possible, if they have a path
        cyclists_not_possible = []
        cyclists = []
        if(len(possible_cyclists) > 0):
            cyclists = self.can_cycle(possible_cyclists)
            cyclists_not_possible = list(set(possible_cyclists) - set(cyclists))

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
                public_transport_unmatched = self.public_transport_matching(users_public_transport, self.bus_users)
                #create actors representing only the buses that are going to pick people up
                for bus_driver in self.bus_users:
                    if(len(bus_driver.users_to_pick_up) > 0):
                        users_turn_actors.append(bus_driver)
            if(service == "bike"):
                users_turn_actors = users_turn_actors + cyclists


        #Make list combining all the users that aren't gonna participate in this run
        users_not_participating = public_transport_unmatched + cyclists_not_possible + ride_sharing_unmatched
        # print("depois de ver se temos stcp ", len(users_turn_actors))

        create_actor_events = self.create_actors_events(users_turn_actors)

        # print("users {}".format(len(self.users)))

        lost = dict()
        lost["sharedCar"] = len(ride_sharing_unmatched)
        lost["bus"] = len(public_transport_unmatched)
        lost["bike"] = len(cyclists_not_possible)

        self.users_lost[self.runn] = lost

        # print("users lost")
        # print(self.users_lost)

        # with open("ughh.txt", 'a+') as f:
        #     print("run ", self.runn, file=f)
        #     print("\n", file=f)
        #     print("ride sharing unmatched  ", len(ride_sharing_unmatched), file=f)
        #     print("public transport unmatched  ", len(public_transport_unmatched), file=f)
        #     print("cycling unmatched  ", len(cyclists_not_possible), file=f)
        #     print("\n", file=f)
        # for user in self.users:
        #     print(user)

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

        for a in self.actors:
            print("actor percurso ")
            print(a.traveled_nodes)
        exit()

        # print("olaaa")
        # for ac in self.actors:
        #     print("eu represento este user: \n", ac.user)
        #     print("esta é a minha rota: \n", ac.base_route)
        #     print("vou buscar: \n", ac.user.users_to_pick_up)

        for actor in self.actors:
            #actor.cost e actor.spent_credits
            #TODO
            #diferenciar entre transporte privado e os outros
            #transporte privado é como está agora
            #transport coletivo:
            #for pelos user_to_pick_up
            # descobrir quando é que o actor chegou ao house_node do user (atraves dos actor.travelled_nodes), pegar no tempo, ver a diferença entre o tempo total e as horas a que ele chegou ao house node
            # mudar assim o actor. travel time e o actor.cost (actor.provider.get_cost(real time))
            # nao esquecer tb do subsidio
            # Ride Sharing:
            # igual ao transport coletivo

            if(actor.service == "car" or actor.service == "bike"):
                commute_out = CommuteOutput(
                    actor.cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
                user_info = dict()
                user_info["user"] = actor.user
                user_info["commute_output"] = commute_out
                user_info["utility"] = actor.user.calculate_utility_value(
                    commute_out)
                #update dos creditos do user
                final_users.append(user_info)
            elif(actor.service == "sharedCar"):
                #create commute output for the driver
                travel_cost = actor.simple_travel_cost()
                travel_cost_portion = travel_cost / (len(actor.user.users_to_pick_up) + 1)
                driver_cost = travel_cost_portion + actor.calculate_transporte_subsidy(len(actor.traveled_nodes)-1)
                commute_out = CommuteOutput(driver_cost, actor.travel_time, actor.awareness, actor.comfort, actor.provider.name)
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
                    rider.time_spent_waiting = max(time_waiting,0)
                    rider_cost = travel_cost_portion + actor.calculate_transporte_subsidy(actor.rider_traveled_dist(rider.house_node))
                    commute_out = CommuteOutput(rider_cost, actor.rider_travel_time(rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
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
                    commute_out = CommuteOutput(actor.rider_cost(rider.house_node), actor.rider_travel_time(
                        rider.house_node) + rider.time_spent_waiting, actor.awareness, actor.comfort, actor.provider.name)
                    user_info = dict()
                    user_info["user"] = rider
                    user_info["commute_output"] = commute_out
                    user_info["utility"] = rider.calculate_utility_value(
                        commute_out)
                    final_users.append(user_info)




            # print("mean: {}  utility: {} ".format(commute_out.mean_transportation, user_info["utility"]))
        # actors_co = self.actors
        # with open("utility_teste.txt", 'w+') as f:
        #     for actor in actors_co:
        #         if(len(actor.user.users_to_pick_up) > 0):
        #             print("sou uma ator de: ", actor.service, "\n", file=f)
        #             print("represento este user: ", actor.user, "\n", file=f)
        #             print("fui buscar estes users: ", actor.user.users_to_pick_up, "\n", file=f)
        #             for user in actor.user.users_to_pick_up:
        #                 print("ele foi-me buscar e eu vivia aqui: ", user.house_node, "\n", file=f)
        #             print("esta foi a minha rota: ", actor.base_route, "\n", file=f)
        #             print("\n", file=f)

        # for user_info in final_users:
        #     with open("utility_teste.txt", 'a+') as f:
        #         print("eu sou o user: ", user_info["user"], "\n", file=f)
        #         print("este foi o meu custo: ", user_info["commute_output"].cost, "\n", file=f)
        #         print("este foi o meu ttt: ",user_info["commute_output"].total_time, "\n", file=f)
        #         print("esta foi a minha utilidade: ", user_info["utility"], "\n", file=f)
        #         print("\n", file=f)


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
        # with open("ughh.txt", 'a+') as f:
        #     print("run ", self.runn, file=f)
        #     self.runn += 1
        #     print("run num final users  ", len(final_users), file=f)

        self.runn += 1
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

            #Choose if the user has a bicycle
            has_bike = 1 if np.random.uniform(
            ) < self.input_config["users"]["has_bike"] else 0

            #Assign the available seats in the private vehicle for each user who has a private vehicle according to the cluster
            if(has_private):
                seats_num = list((self.input_config["users"]["clusters"][chosen_cluster]["seat_probs"]).keys())
                seats_percentages = (self.input_config["users"]["clusters"][chosen_cluster]["seat_probs"]).values()

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
                        chosen_course, chosen_grade, salary, budget, available_seats, distance_from_destination, bool(has_bike))

            # se estiverem entao proximo passo é adicionar tambem informaçao de ano e curso!
            users.append(user)

        return users
