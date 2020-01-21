from vehicles_info import Car, Bus
from utils import bidict

providers = bidict({
    "Personal": 0,
    "Friends": 1,
    "STCP": 2
})

fuel_cost = 1/0.6

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
        return time*fuel_cost + fixed_added_cost
    
    def get_time(self, time):
        return time

    def get_comfort(self):
        return 1

    def get_emissions(self, time):
        return Car.emissions(time)

    def get_awareness(self):
        return Car.awareness()

class Friends(Provider):

    n_passengers = 2
    def __init__(self):
        super().__init__("Friends", "sharedCar")
    
    
    def get_cost(self,time):
        return (time*fuel_cost + fixed_added_cost)/self.n_passengers
    
    def get_time(self, time):
        return time + 1

    def get_comfort(self):
        return 0.6

    def get_emissions(self, time):
        return Car.emissions(time)/self.n_passengers

    def get_awareness(self):
        return 0.5

class STCP(Provider):
    def __init__(self):
        super().__init__("STCP", "bus")
    
    def get_cost(self,time):
        return 2
    
    def get_time(self, time):
        return time + 3

    def get_comfort(self):
        return 0.4

    def get_emissions(self, time):
        return Bus.emissions(time)

    def get_awareness(self):
        return Bus.awareness()
