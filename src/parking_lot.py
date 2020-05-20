class ParkingLot():

    lot_capacity: int
    lot_capacity_shared: int
    available_seats_shared: int
    available_seats_personal: int
    parking_cost_shared: float
    parking_cost_personal: float



    PARKING_OUTSIDE: float

    def __init__(self,lot_capacity:int, lot_capacity_shared: int, parking_cost_personal: float, parking_cost_shared: float):
        self.lot_capacity = lot_capacity
        self.lot_capacity_shared = lot_capacity_shared
        self.available_seats_personal = lot_capacity - lot_capacity_shared
        self.available_seats_shared = lot_capacity_shared
        self.parking_cost_personal = parking_cost_personal
        self.parking_cost_shared = parking_cost_shared

    def get_parking_cost(self, service:str):
        #if the parking lot has enough space
        if(self.check_parking_spot(service)):
            if(service == "car"):
                return self.parking_cost_personal
            elif(service == "sharedCar"):
                return self.parking_cost_shared
        else:
            return self.PARKING_OUTSIDE
        

    def add_vehicle(self,service:str):
        if(service == "car"):
            self.available_seats_personal -= 1
        elif(service == "sharedCar"):
            self.available_seats_shared -= 1

    def remove_vehicle(self,service:str):
        if(service == "car"):
            self.available_seats_personal += 1
        elif(service == "sharedCar"):
            self.available_seats_shared += 1

    def check_parking_spot(self,service: str):
        if(service == "car"):
            return (self.available_seats_personal > 0)
        if(service == "sharedCar"):
            return (self.available_seats_shared > 0)

    def park_vehicle(self,service: str):
        #if the parking lot has available spaces
        if(self.check_parking_spot(service)):
            cost = self.get_parking_cost(service)
            self.add_vehicle(service)
            return cost
        #have to park outside
        else:
            cost = self.get_parking_cost(service)
            return cost
