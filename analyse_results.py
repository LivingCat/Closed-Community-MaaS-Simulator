import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

filename = "results/normal_100_emissions.npy"

results_dict = np.load(
    filename, allow_pickle=True)
results_dict = results_dict.item()
# print(results_dict["total"][-1])
# print(results_dict["bus"][-1])
# print(results_dict["car"][-1])
# print(results_dict["sharedCar"][-1])
del results_dict["total"]
mean_total = 0
for key in results_dict:
    if key != "total":
        mean_total += np.mean(results_dict[key][-200:])

for key in sorted(results_dict):
    print("{} final: {} ({}%)".format(key, np.mean(results_dict[key][-200:]), np.mean(results_dict[key][-200:])*100/mean_total))

for key in results_dict:
    results_dict[key] = savgol_filter(results_dict[key], 101,1)
    results_dict[key] = results_dict[key][:2900]

x = np.arange(3)

for i in range(len(results_dict["car"])):
    current = []
    for key in results_dict:
        current.append(results_dict[key][i])
    plt.ylim((2000, 12500))
    plt.ylabel("emissions CO2 (g)")
    plt.bar(x,current)
    plt.xticks(x, ('car', 'bus', 'shared car'))
    plt.savefig("100_emissions_gif/frame{}".format(i))
    plt.clf()
    print(i)

# for key in sorted(results_dict):
#     plt.plot(results_dict[key][:2900], label=key)

# plt.plot(results_dict["car"], label="no smooth")
# plt.plot(savgol_filter(results_dict["car"], 101, 2), label="smooth")
# plt.legend(loc='upper left')
# plt.xlabel("iterations")
# if "number" in filename:
#     plt.ylabel("number of users")
# else:
#     plt.ylabel("emissions CO2 (g)")
# filename = filename.replace("results", "final_results")
# filename = filename.replace(".npy", ".png")
# plt.savefig(filename)
# plt.show()
