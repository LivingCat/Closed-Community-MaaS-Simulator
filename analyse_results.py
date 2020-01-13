import numpy as np
import matplotlib.pyplot as plt
results_dict = np.load("cheaper_bus_100_emissions.npy", allow_pickle=True)
results_dict = results_dict.item()
print(results_dict["total"][-1])
print(results_dict["bus"][-1])
print(results_dict["car"][-1])
print(results_dict["sharedCar"][-1])