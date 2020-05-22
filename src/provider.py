from vehicles_info import Car, Bus, Bike, Walk
from utils import bidict

providers = bidict({
    "Personal": 0,
    "Friends": 1,
    "STCP": 2
})

car_cost = 2*0.36
bus_cost = 2*0.11
friend_cost_2 = 2 * 0.144
friend_cost_more = 2 * 0.11

fixed_added_cost = 0

class Provider:
    """Represents an provider/operator - one of the stakeholders of the system"""
    def __init__(self,name,service):
        self.name = name
        self.service = service


class Personal(Provider):
    def __init__(self):
        super().__init__("Personal", "car")
    
    def get_cost(self,time):
        return time* Car.speed * car_cost + fixed_added_cost
    
    def get_time(self, time):
        return time

    def get_comfort(self):
        return 1

    def get_emissions(self, time):
        return Car.emissions(time)

    def get_awareness(self):
        return Car.awareness()
    
    def get_credits(self):
        return Car.credits()
    
    def get_speed(self):
        return Car.speed

class Friends(Provider):

    def __init__(self):
        super().__init__("Friends", "sharedCar")
    
    
    def get_cost(self,time,num_riders):
        # - creditos/min_creditos_para_desconto
        # if(num_riders == 1):
        #     return time * Car.speed * friend_cost_2 + fixed_added_cost
        # else:
        #     return time * Car.speed * friend_cost_more + fixed_added_cost

        return time * Car.speed * car_cost + fixed_added_cost
    
    def get_time(self, time):
        return time

    def get_comfort(self):
        return 0.6

    def get_emissions(self, time):
        return Car.emissions(time)

    def get_awareness(self):
        return 0.5

    def get_credits(self):
        return Car.credits() + 1


class STCP(Provider):
    def __init__(self):
        super().__init__("STCP", "bus")
    
    def get_cost(self,time):
        return time * Bus.speed * bus_cost
    
    def get_time(self, time):
        return time

    def get_comfort(self):
        return 0.4

    def get_emissions(self, time):
        return Bus.emissions(time)

    def get_awareness(self):
        return Bus.awareness()

    def get_credits(self):
        return Bus.credits()


class Bicycle(Provider):
    def __init__(self):
        super().__init__("Bicycle", "bike")

    def get_cost(self, time):
        return 0

    def get_time(self, time):
        return time

    def get_comfort(self):
        return 0.5

    def get_emissions(self, time):
        return Bike.emissions(time)

    def get_awareness(self):
        return Bike.awareness()

    def get_credits(self):
        return Bike.credits()


class Walking(Provider):
    def __init__(self):
        super().__init__("Walking", "walk")

    def get_cost(self, time):
        return 0

    def get_time(self, time):
        return time

    def get_comfort(self):
        return 0.5

    def get_emissions(self, time):
        return Walk.emissions(time)

    def get_awareness(self):
        return Walk.awareness()

    def get_credits(self):
        return Walk.credits()
