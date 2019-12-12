class Personality:
    """Represents the user's preferences of the system"""

    def __init__(self, willingness_to_pay, willingness_to_wait, awareness, has_private, mean_transportation=""):
        self.willingness_to_pay = willingness_to_pay
        self.willingness_to_wait = willingness_to_wait
        self.mean_transportation = mean_transportation
        self.awareness = awareness
        self.has_private = has_private


class CommuteOutput:
     def __init__(self, cost, total_time, awareness, mean_transportation):
        self.cost = cost
        self.total_time = total_time
        self.awareness = awareness
        self.mean_transportation = mean_transportation


class User:
    """Represents a User of the system"""

    
    def __init__(self, personality: Personality, start_time):
       self.personality = personality
       self.start_time = start_time

    def cost_util(self, commute_out: CommuteOutput):
        return commute_out.cost * self.personality.willingness_to_pay

    def time_util(self, commute_out: CommuteOutput):
        return commute_out.total_time * self.personality.willingness_to_wait

    def social_util(self, commute_out: CommuteOutput):
        return commute_out.awareness * self.personality.awareness


    def calculate_utility_value(self, commute_out: CommuteOutput):
        return self.cost_util(commute_out) + self.time_util(commute_out) + self.social_util(commute_out)


    
