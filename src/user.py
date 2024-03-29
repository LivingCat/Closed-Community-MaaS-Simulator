from provider import Provider
from typing import List
class Personality:
    """Represents the user's preferences of the system"""

    def __init__(self, willingness_to_pay: float, willingness_to_wait: float, awareness: float, comfort_preference: float, dim1: float, dim2: float, dim3: float, dim4: float, dim5: float, dim6:float, mean_transportation=""):
        self.willingness_to_pay = willingness_to_pay
        self.willingness_to_wait = willingness_to_wait
        self.awareness = awareness
        self.comfort_preference = comfort_preference
        self.mean_transportation = mean_transportation

        #Factors
        #self.friendliness = friendliness
        #self.suscetible = suscetible
        #self.transport = transport
        #self.urban = urban
        #self.willing = willing
        self.dim1 = dim1
        self.dim2=dim2
        self.dim3=dim3
        self.dim4=dim4
        self.dim5=dim5
        self.dim6=dim6


class CommuteOutput:
    def __init__(self, cost, total_time, awareness, comfort, mean_transportation):
        self.cost = cost
        self.total_time = total_time
        self.awareness = awareness
        self.mean_transportation = mean_transportation
        self.comfort = comfort

    def __str__(self):
        return "Cost %s, total time %s, awareness %s, mean transportation %s , comfort %s" % (self.cost, self.total_time, self.awareness, self.mean_transportation, self.comfort)


class User:
    """Represents a User of the system"""
    provider: Provider
    mean_transportation: str
    cluster: str
    course: str
    grade: str
    salary: float
    budget: float
    friends: List['User']
    num_friends: int
    available_seats: int
    distance_from_destination: int
    house_node: int
    users_to_pick_up: List['User']
    route_name: str
    route: List[int]
    schedule: {}
    capacity: int
    time_spent_waiting: float
    has_bike: bool
    credits_own: float
    credits_spent: float
    credits_won_round: float
    can_cycle: bool
    can_walk: bool
  

    static_model_id = 0
    # static_model_id = 1


    def __init__(self, personality: Personality, start_time: float, cluster: str, course: str, grade:str, salary: float, budget: float, available_seats: int, distance_from_destination: int, has_bike: bool, has_private: bool):
       self.personality = personality
       self.start_time = start_time
       self.cluster = cluster
       self.course = course
       self.grade = grade
       self.salary = salary
       self.budget = budget
       self.friends = list()
       self.available_seats = available_seats
       self.capacity = available_seats
       self.distance_from_destination = distance_from_destination
       self.num_friends = 0
       self.users_to_pick_up = []
       self.time_spent_waiting = 0.0
       self.has_bike = has_bike
       self.has_private = has_private
       self.credits_own = 10
       self.credits_spent = 0.0
       self.credits_won_round = 0.0

       self.can_cycle = False
       self.can_walk = False

       self.friends_ids = []


       """"Initializer for the ID of all user models"""
       self.user_id = User.static_model_id
       User.static_model_id += 1

    @staticmethod
    def default():
        return User(Personality(0,0,0,0,0,0,0,0,0,0), 0.0, "","","",0.0,0.0,0,0,False,False)

    def my_copy(self, service: str):
        new_user = User.default()
        if(service == "bus"):
            new_user.house_node = -1
        else:
            new_user.house_node = self.house_node
        new_user.house_nodes_riders = []
        new_user.riders_num = len(self.users_to_pick_up)
        
        for rider in self.users_to_pick_up:
            new_user.house_nodes_riders.append(rider.house_node)
        return new_user

    def set_route_name(self,route_name: str):
        self.route_name = route_name

    def set_route(self, route: List[int]):
        self.route = route

    def add_friends(self,friends: List['User']):
        self.friends = friends

    def add_house_node(self, house_node: int):
        self.house_node = house_node

    def cost_util(self, commute_out: CommuteOutput):
        max_cost = 15
        # if(commute_out.cost == 0):
        #     return 0
        # else:
        #     #normalize 
        #     return (1 - (commute_out.cost/max_cost)) * self.personality.willingness_to_pay
            # return 1/(commute_out.cost) * self.personality.willingness_to_pay
        normalize_cost = commute_out.cost/max_cost
        # return (-1/(self.personality.willingness_to_pay * 10))* (normalize_cost**2) + 1
        # return (-1/(self.personality.willingness_to_pay * 5))* (normalize_cost**2) + 1
        # return ((-1/(self.personality.willingness_to_pay * 2)) * normalize_cost + 1)
        # print("will pay", self.personality.willingness_to_pay)
        # print("cost util", ((-1/(self.personality.willingness_to_pay * 5))
        #                     * normalize_cost + 1))
        return ((-1/(self.personality.willingness_to_pay * 5)) * normalize_cost + 1)
        # return ((-1/(self.personality.willingness_to_pay * 10)) * normalize_cost + 1)

    def time_util(self, commute_out: CommuteOutput):
        max_time = 0.70
        #normalize
        # return (1 - (commute_out.total_time/max_time)) * self.personality.willingness_to_wait
        # return 1/(commute_out.total_time) * self.personality.willingness_to_wait
        normalize_time = commute_out.total_time/max_time
        # return (-1/(self.personality.willingness_to_wait * 10)) * (normalize_time**2) + 1
        # return (-1/(self.personality.willingness_to_wait * 5)) * (normalize_time**2) + 1
        # return ((-1/(self.personality.willingness_to_wait * 2)) * normalize_time + 1)
        return ((-1/(self.personality.willingness_to_wait * 5)) * normalize_time + 1)
        # return ((-1/(self.personality.willingness_to_wait * 10)) * normalize_time + 1)

    def social_util(self, commute_out: CommuteOutput):
        return commute_out.awareness * self.personality.awareness

    def comfort_util(self, commute_out: CommuteOutput):
        return commute_out.comfort * self.personality.comfort_preference

    def calculate_utility_value(self, commute_out: CommuteOutput):
        return self.cost_util(commute_out) + self.time_util(commute_out) + self.social_util(commute_out) + self.comfort_util(commute_out)

    def get_user_current_state(self):
        personality = self.personality
        return [self.start_time, personality.willingness_to_pay, personality.willingness_to_wait, personality.awareness, personality.comfort_preference, int(self.has_private), int(self.has_bike), self.credits_own, self.distance_from_destination, self.capacity, self.can_cycle, self.can_walk]
    
    def add_credits(self,credits_gained:int):
        self.credits_own += credits_gained
        self.credits_won_round = credits_gained
        return self.credits_own

    def remove_credits(self,credits_spent:int):
        self.credits_own -= credits_spent
        self.credits_spent += credits_spent

    def credits_discount(self, min_credits: int, credit_value: float):
        #User has the minimum amount of credits which makes him elegible for discount
        discount = -1
        if(self.credits_own >= min_credits):
            discount = min_credits * credit_value
            #remove credits spent
            self.remove_credits(min_credits)
        return discount
        
    def get_social_credits(self, percent: float):
        if(self.mean_transportation == "car" or self.mean_transportation == "bike" or self.mean_transportation == "walk"):
            print("am either car or bike or walk")
            return 0
        else:
            soc_cre = 0
            for friend in self.friends:
                c = friend.credits_won_round * percent
                #user gains credits from his friend
                soc_cre += c
                #friend loses credits he gave away
                friend.give_social_credits(c)
            return soc_cre                


    def add_social_credits(self,soc_credits:float):
        self.credits_own += soc_credits
        return self.credits_own

    def give_social_credits(self, soc_credits:float):
        self.credits_own -= soc_credits


    def __str__(self):
        return "I am %s, I live here %s, cluster %s, have private %s, have bike %s , credits i own %s, credits i spent %s \n" % (self.user_id,self.house_node, self.cluster, self.has_private, self.has_bike, self.credits_own, self.credits_spent)

