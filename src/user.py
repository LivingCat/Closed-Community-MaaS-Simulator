from provider import Provider
from typing import List
class Personality:
    """Represents the user's preferences of the system"""

    def __init__(self, willingness_to_pay, willingness_to_wait, awareness, comfort_preference, has_private, friendliness, suscetible, transport, urban, willing, mean_transportation=""):
        self.willingness_to_pay = willingness_to_pay
        self.willingness_to_wait = willingness_to_wait
        self.awareness = awareness
        self.comfort_preference = comfort_preference
        self.has_private = has_private
        self.mean_transportation = mean_transportation

        #Factors
        self.friendliness = friendliness
        self.suscetible = suscetible
        self.transport = transport
        self.urban = urban
        self.willing = willing


class CommuteOutput:
     def __init__(self, cost, total_time, awareness, comfort, mean_transportation):
        self.cost = cost
        self.total_time = total_time
        self.awareness = awareness
        self.mean_transportation = mean_transportation
        self.comfort = comfort


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

    def __init__(self, personality: Personality, start_time: float, cluster: str, course: str, grade:str, salary: float, budget: float, available_seats: int):
       self.personality = personality
       self.start_time = start_time
       self.cluster = cluster
       self.course = course
       self.grade = grade
       self.salary = salary
       self.budget = budget
       self.friends = list()
       self.available_seats = available_seats
       self.num_friends = 0

    def add_friends(self,friends: List['User']):
        self.friends = friends

    def cost_util(self, commute_out: CommuteOutput):
        return 1/(commute_out.cost) * self.personality.willingness_to_pay

    def time_util(self, commute_out: CommuteOutput):
        return 1/(commute_out.total_time) * self.personality.willingness_to_wait

    def social_util(self, commute_out: CommuteOutput):
        return commute_out.awareness * self.personality.awareness

    def comfort_util(self, commute_out: CommuteOutput):
        return commute_out.comfort * self.personality.comfort_preference

    def calculate_utility_value(self, commute_out: CommuteOutput):
        return self.cost_util(commute_out) + self.time_util(commute_out) + self.social_util(commute_out) + self.comfort_util(commute_out)

    def get_user_current_state(self):
        personality = self.personality
        return [self.start_time, personality.willingness_to_pay, personality.willingness_to_wait, personality.awareness, personality.comfort_preference, int(personality.has_private)]
    
    def pprint(self):
        personality = self.personality
        print("cluster: {} \n". format(self.cluster))
        print("course: {} \n". format(self.course))
        print("grade: {} \n". format(self.grade))
        print("salary: {} \n". format(self.salary))
        print("budget: {} \n". format(self.budget))
        print("willingness to pay: {} \n". format(personality.willingness_to_pay))
        # print("willingness to wait: {} \n". format(personality.willingness_to_wait))
        # print("awareness: {} \n". format(personality.awareness))
        # print("comfort preference: {} \n". format(personality.comfort_preference))
        # print("friendliness: {} \n". format(personality.friendliness))
        # print("suscetible: {} \n". format(personality.suscetible))
        # print("transport: {} \n". format(personality.transport))
        # print("urban: {} \n". format(personality.urban))
        # print("willing: {} \n". format(personality.willing))
        print("friends: {} \n".format(self.friends))
        print("has private: {} \n". format(personality.has_private))
        print("available seats: {} \n".format(self.available_seats))
        return True

