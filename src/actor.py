"""
Class to represent an Actor, its information, decision-making process, and behaviour.
"""
from typing import List, Tuple
from collections import defaultdict
from user import User
from provider import Provider
import copy

class Actor():
    
    TIME_INDEX = 0
    NODE_INDEX = 1

    base_route: List[int]

    # Tuple[time actor arrived to node, node]
    traveled_nodes: List[Tuple[float, int]]

    route_travel_time: defaultdict
    total_travel_time: float

    static_model_id = 1

    parking_cost = 0

    def __init__(self, route: List[int], user: User, provider: Provider):
        self.user = user
        self.base_route = route
        self.provider = provider
        self.service = self.provider.service
        self.route_travel_time = defaultdict(lambda: 0.0)
        self.total_travel_time = 0.0
        self.start_time = 0.0
        """"Initializer for the ID of all actor models"""
        self.actor_id = Actor.static_model_id
        Actor.static_model_id += 1

    def __repr__(self):
        return "A%d :: TI %.4f :: TTT %.4f" % (self.actor_id, round(self.start_time, 4), round(self.total_travel_time, 4))

    def my_copy(self):
        new_actor = Actor(self.base_route, self.user,self.provider)
        # new_actor.user = copy.deepcopy(self.user)
        new_actor.user = self.user.my_copy(self.provider.service)

        new_actor.route_travel_time = self.route_travel_time
        new_actor.total_travel_time = self.total_travel_time
        new_actor.start_time = self.start_time
        new_actor.traveled_nodes = self.traveled_nodes
        return new_actor
        

    def add_time_for_edge(self, edge: Tuple[int, int], tt: float):
        self.route_travel_time[edge] = tt

    def update_total_tt(self):
        self.total_travel_time = sum(self.route_travel_time.values())

    def reached_dest(self) -> bool:
        """Check whether this actor reached its destination"""
        return self.base_route[-1] == self.traveled_nodes[-1][self.NODE_INDEX]

    def get_next_travel_edge(self, timestamp: float) -> Tuple[int, int]:
        """Gets the next edge to be traveled"""
        return (self.traveled_nodes[-1][self.NODE_INDEX],
                self.base_route[len(self.traveled_nodes)])

    def start_trip(self, at_time: float):
        """Make the actor start the route, at the given time"""
        self.start_time = at_time
        self.traveled_nodes = [(at_time, self.base_route[0])]   # Start node

    def travel(self, at_time: float, edge: Tuple[int, int]):
        """Makes the Actor travel the given edge"""
        if not self.traveled_nodes[-1][self.NODE_INDEX] == edge[0]:
            raise Exception

        self.traveled_nodes.append((at_time, edge[1]))

    def print_traveled_route(self):
        """Pretty printing of the Actor's traveled route"""
        print("Actor #%d:" % self.actor_id)
        print("\tNode: %d, timestamp: %f" %
              (self.traveled_nodes[0][self.NODE_INDEX],
               self.traveled_nodes[0][self.TIME_INDEX]))

        for i in range(1, len(self.traveled_nodes)):
            print("\tNode: %d, timestamp: %f (+%f)" %
                  (self.traveled_nodes[i][self.NODE_INDEX],
                   self.traveled_nodes[i][self.TIME_INDEX],
                   self.traveled_nodes[i][self.TIME_INDEX]
                   - self.traveled_nodes[i-1][self.TIME_INDEX]))

    @property
    def cost(self):
        #incorporate transport subsidy for each of the users that this actor represents (in simulator, commute output)
        # transport subsidy depends on the mode of transport used:
        # 0.36€/km for private transport
        # 0.11€/km for public transport
        return (self.provider.get_cost(self.total_travel_time) - self.calculate_transporte_subsidy(len(self.traveled_nodes)-1))

    def sharing_travel_cost(self):
        return self.provider.get_cost(self.total_travel_time, len(self.user.users_to_pick_up))


    def rider_travel_time(self,house_node: int):
        time_reached_house = 0.0
        for time,node in self.traveled_nodes:
            if (node == house_node):
                time_reached_house = time
                break
        time_reached_dest = self.traveled_nodes[-1][0]
        
        return (time_reached_dest - time_reached_house)
            
    def driver_reached_pick_up(self,house_node: int):
        time_reached_house = 0.0
        for time,node in self.traveled_nodes:
            if (node == house_node):
                time_reached_house = time
                break
        return time_reached_house


    def rider_traveled_dist(self,house_node: int):
        rider_index = self.base_route.index(house_node) + 1
        return (len(self.base_route) - rider_index )

    def rider_cost(self,house_node: int):
        rider_tt = self.rider_travel_time(house_node)
        rider_trav_dist = self.rider_traveled_dist(house_node)
        # print("custo total ", self.provider.get_cost(rider_tt))
        # print("subsidio ", self.calculate_transporte_subsidy(rider_trav_dist))
        return self.provider.get_cost(rider_tt) - self.calculate_transporte_subsidy(rider_trav_dist)

    @property
    def travel_time(self):
        return self.provider.get_time(self.total_travel_time)
    
    @property
    def awareness(self):
        return self.provider.get_awareness()

    @property
    def comfort(self):
        return self.provider.get_comfort()

    @property
    def emissions(self):
        return self.provider.get_emissions(self.total_travel_time)

    def emissions_bus(self,tt:float):
        return self.provider.get_emissions(tt)


    def calculate_transporte_subsidy(self, num_travelled_nodes: int):
        # traveled nodes has all the nodes the actor travelled through, minus 1 gives us the number of edges he used and consequently the km travelled
        if(self.service == "car"):
            return 0.36 * num_travelled_nodes
        elif(self.service == "bus"):
            return 0.11 * num_travelled_nodes
        #need to change the value for shared car
        elif(self.service == "sharedCar"):
            if(len(self.user.users_to_pick_up) == 1):
                return 0.144 * num_travelled_nodes
            else:
                return 0.11 * num_travelled_nodes
        elif(self.service == "bike" or self.service=="walk"):
            return 0

    def add_parking_cost(self,cost:float):
        self.parking_cost = cost
    
    def get_parking_cost(self):
        return self.parking_cost



