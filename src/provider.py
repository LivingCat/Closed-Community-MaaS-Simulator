from vehicles_info import Car, Bus
from utils import bidict

providers = bidict({
    "Personal": 0,
    "Friends": 1,
    "STCP": 2
})


class Provider:
    """Represents an provider/operator - one of the stakeholders of the system"""
    def __init__(self,name,service):
        self.name = name
        self.service = service


class Personal(Provider):
    def __init__(self):
        super().__init__("Personal", "car")
    
    def actor_factory():
        return Actor(route, user, self)
    
    def get_cost(self,time):
        return time*4
    
    def get_time(self, time):
        return time

    def get_comfort(self):
        return 0.8

    def get_emissions(self, distance):
        return Bus.emissions(distance)

    def get_awareness(self):
        return Bus.awareness()

class Friends(Provider):
    def __init__(self):
        super().__init__("Friends", "sharedCar")
    
    def actor_factory():
        return Actor(route,user,self)
    
    def get_cost(self,time):
        return (time*4)/3
    
    def get_time(self, time):
        return time

    def get_comfort(self):
        return 0.6

    def get_emissions(self, distance):
        return Car.emissions(distance)/3

    def get_awareness(self):
        return 0.5

class STCP(Provider):
    def __init__(self):
        super().__init__("STCP", "bus")
    
    def actor_factory():
        return Actor(route, user, self)
    
    def get_cost(self,time):
        return 2
    
    def get_time(self, time):
        return time + 2

    def get_comfort(self):
        return 0.4

    def get_emissions(self, distance):
        return Bus.emissions(distance)

    def get_awareness(self):
        return Bus.awareness()
