"""
Main project source file.
"""
from typing import List, Tuple
from ipdb import set_trace
from actor import Actor
from data_plotting import plot_accumulated_actor_graph, plot_accumulated_edges_graphs, plot_emissions_development, plot_number_users_development
from simulator import Simulator
from provider import Provider, Personal, Friends, STCP, Bicycle
from graph import RoadGraph
from user import User, Personality
from queue import PriorityQueue
from utils import softmax_travel_times, compute_average_over_time, MultimodalDistribution, get_time_from_traffic_distribution
from statistics import SimStats
from DeepRL import DQNAgent
from pprint import pprint
from collections import defaultdict
from tqdm import trange
from functools import partial


import argparse
import numpy as np
import json
import os.path
import networkx as nx
import matplotlib.pyplot as plt
import json
import time
import copy

import csv

run_name = "test_average_occupancy"
CARBON_TAX = 180.0

def parse_args():
    parser = argparse.ArgumentParser(
        description='Systems Modelling and Simulation')

    parser.add_argument("-js", "--json", default="input.json", type=str, dest='json_file', metavar="JSON", help="input configuration file")

    parser.add_argument("-n", "--num_actors", default=500, type=int, metavar="N",
                        help="number of vehicles/actors to generate per simulation run")

    parser.add_argument("-r", "--runs", default=1, type=int, metavar="R", dest="n_runs",
                        help="number of times to run the simulation for")

    parser.add_argument("-thr", "--congestion_threshold", default=0.9, type=float, metavar="THRESH",
                        help="threshold when to consider a link congested, volume/capacity")

    parser.add_argument("-tmax", "--max_run_time", default=48.0, type=float, metavar="MAX_TIME",
                        dest="max_run_time", help="max time of each simulation run (in hours)")

    parser.add_argument("-p", "--peak", type=float, nargs=2, action='append',
                        dest='traffic_peaks', metavar=("TPEAK_MEAN", "TPEAK_STD"),
                        help="mean and standard deviation of a normal distribution that represents a peak in traffic")

    parser.add_argument("-o", "--out_file", type=str, default=os.path.join("src", "results", "default.json"),
                        dest='save_path', metavar="SAVE_PATH",
                        help="place to save the result of running the simulations")

    parser.add_argument("-v", "--verbose", dest='verbose', action="store_true",
                        help="allow helpful prints to be displayed")
    parser.set_defaults(verbose=False)

    parser.add_argument("-pl", "--plots", dest='plots', action="store_true",
                        help="display plots at the end of the simulation regarding the network occupation")
    parser.set_defaults(plots=True)

    return parser.parse_args()


def print_args(args):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print()


def actor_constructor(graph: RoadGraph, user: User):
    """Calculate possible routes and give each one a probability based on how little time it takes to transverse it"""
    #When ride sharing, limiting the possible routes for the ones that pass through the house nodes of the riders

    #if we are in the presence of the default bus user
    if(user.provider.service == "bus"):
        return Actor(user.route, user, user.provider)

    possible_routes = graph.get_all_routes(user.house_node,user.mean_transportation)
      
    actual_poss_routes = [] 
    
    if(len(user.users_to_pick_up) > 0):
        for route in possible_routes:
            all_in= True
            for rider in user.users_to_pick_up:
                if(rider.house_node not in route):
                    all_in = False
                    break
            if(all_in):
                actual_poss_routes.append(route)
    else:
        actual_poss_routes = possible_routes
    routes_times = [graph.get_optimal_route_travel_time(r)
                    for r in actual_poss_routes]
    routes_probs = softmax_travel_times(routes_times)
    idx = np.random.choice(len(actual_poss_routes), p=routes_probs)

    return Actor(actual_poss_routes[idx], user, user.provider)



def stats_constructor(graph: RoadGraph):
    return SimStats(graph)


def statistics_print(sim: Simulator):
    """Print of simulation statistics regarding non ATIS users"""
    print()
    atis_no = []
    for a in sim.actors:
        atis_no.append(a.total_travel_time)

    print("ATIS NO: mean: %f || std: %f" % (np.mean(atis_no), np.std(atis_no)))


def average_all_results(all_s: List[SimStats], display_plots: bool, users_lost: dict(dict()), time_per_mode_last_runs: dict()):

    #Get Min Users lost run
    min_val = 99999999
    min_run = -1
    last_runs = 100

    # print("users lost")
    # print(users_lost)

    for key in users_lost.keys():
        total = users_lost[key]["total"]
        if(total < min_val):
            min_val = total
            min_run = key


    """Gather information regarding all runs and its metrics"""
    
    print("tou no average_all_results \n")

    max_emissions = 0
    max_carbon_tax = 0
    max_transport_subsidy = 0
    max_combined_cost = 0


    # gather summary information
    actors_wo_end = [
        len([1 for a in stats.actors if not a.reached_dest()]) for stats in all_s]
    avg_actors_not_finishing = np.sum(actors_wo_end) / len(all_s)

    actors_summary = [compute_average_over_time(
        stats.actors_in_graph) for stats in all_s]
    edges_summary = [{str(e): compute_average_over_time(stats.edges_flow_over_time[e])
                      for e in stats.edges_flow_over_time} for stats in all_s]

    # gather atis information
    atis_no = np.hstack(
        [[a.total_travel_time for a in stats.actors] for stats in all_s])

    results = {'avg_actors_not_finishing': avg_actors_not_finishing,
               'avg_actors': [np.mean(actors_summary), np.std(actors_summary)],
               'avg_edges': defaultdict(lambda: []),
               'time_atis_no': [np.mean(atis_no), np.std(atis_no)] if len(atis_no) > 0 else [np.nan, np.nan]}

    for d in edges_summary:
        for d_k in d:
            results['avg_edges'][d_k].append(d[d_k])

    results['avg_edges'] = {
        e: [np.mean(results['avg_edges'][e]), np.std(results['avg_edges'][e])] for e in results['avg_edges']
    }

    # gather new information with atis separation
    actors_flow = defaultdict(lambda: [(0.0, 0)])

    print("resultados iniciais")

    for s in all_s:
        for key in s.actors_atis.keys():
            s.actors_atis[key] = s.actors_atis[key][1:]
            for tuple in s.actors_atis[key]:
                actors_flow[key].append(tuple)

    for key in actors_flow.keys():
        actors_flow[key] = sorted(actors_flow[key], key=lambda t: t[0])

    actors_flow_acc = defaultdict(lambda: [(0.0, 0)])

    for key in actors_flow.keys():
        for actor_tuple in actors_flow[key]:
            actors_flow_acc[key].append([actor_tuple[0],
                                        actor_tuple[1] + actors_flow_acc[key][-1][1]])

    # plot_accumulated_actor_graph(actors_flow_acc, len(all_s))
    # plt.waitforbuttonpress(0)

    print("flow")
    results['actors_atis_natis'] = actors_flow_acc

    # the above but for every edge
    # inner_default_dict = lambda: defaultdict(lambda: [])
    # results['edges_occupation'] = defaultdict(inner_default_dict)

    # for s in all_s:
    #     edges = s.edges_flow_atis
    #     for key in edges.keys():
    #         for service in edges[key]:
    #             results['edges_occupation'][str(
    #                 key)][service].append(edges[key][service])

    # with open("edges_occupation.txt", 'w+') as f:
    #     print(results['edges_occupation'], file=f)

    # print("vou agora para os results[edges ocupation] \n")
    # for e_key in results['edges_occupation'].keys():
    #     edge_flow = defaultdict(lambda: [(0.0, 0)])
    #     # pretty(results['edges_occupation'][e_key])
    #     for actor_key in results['edges_occupation'][e_key]:
    #         for tuple_list in results['edges_occupation'][e_key][actor_key]:
    #             tuple_list = tuple_list[1:]
    #             for tuple in tuple_list:
    #                 edge_flow[actor_key].append(tuple)

    #     print("vou para o edge_flow.keys")

    #     for key in edge_flow.keys():
    #         edge_flow[key] = sorted(edge_flow[key], key=lambda t: t[0])

    #     edge_flow_acc = defaultdict(lambda: [(0.0, 0)])

    #     print("vou para o edge_flow.keys outra vez")
    #     for actor_key in edge_flow.keys():
    #         for edge_tuple in edge_flow[actor_key]:
    #             edge_flow_acc[actor_key].append([edge_tuple[0],
    #                                 edge_tuple[1] + edge_flow_acc[actor_key][-1][1]])

    #     print("vou para o edge flow acc keys")
    #     for actor_key in edge_flow_acc.keys():
    #         edge_flow_acc[actor_key] = edge_flow_acc[actor_key][1:]

    #     # print("acc")
    #     # print(edge_flow_acc)
    #     results['edges_occupation'][e_key] = edge_flow_acc

    # print("vou calcular as emissoes e os users \n")

    emissions_dict={
        "car":[],
        "bus":[],
        "sharedCar":[],
        "bike":[],
        "total":[]
    }


    number_users_dict = {
        "car": [],
        "bus": [],
        "sharedCar": [],
        "bike": [],
        "total":[]
    }

    number_actors_dict = {
        "car": [],
        "bus": [],
        "sharedCar": [],
        "bike": [],
        "total": []
    }

    transport_subsidy_dict = {
        "car": [],
        "bus": [],
        "sharedCar": [],
        "total": []
    }

    tax_list = []

    runn = 0

    # if(actor.service != "bus"):
    #     average_tt[actor.service] += actor.user.CommuteOutput

    print("vou fazer emissions, num actor e num users")
    for run in all_s:

        # print("num actors: ", len(run.actors))
        # users = 0
        # for a in run.actors:
        #     users += len(a.user.users_to_pick_up)
        # print("num users: ", users)


        run_emissions_dict={
            "car": 0,
            "bus" :0,
            "sharedCar": 0,
            "bike": 0,
            "total": 0
        }
        run_number_users_dict = {
            "car": 0,
            "bus" :0,
            "sharedCar": 0,
            "bike": 0,
            "total":0
        }

        run_number_actors_dict = {
            "car": 0,
            "bus": 0,
            "sharedCar": 0,
            "bike": 0,
            "total": 0
        }

        run_transport_subsidy_dict = {
            "car": 0,
            "bus": 0,
            "sharedCar": 0,
            "total": 0
        }


        # with open("ughh.txt", 'a+') as f:
        #     print("run ", runn, file=f)
        #     runn += 1
        #     print("run actors number ", len(run.actors), file=f)
        #     for actor in run.actors:
        #         if(len(actor.user.users_to_pick_up) > 0):
        #             print("sou uma ator de: ", actor.service, "\n", file=f)
        #             # print("represento este user: ", actor.user, "\n", file=f)
        #             print("fui buscar estes users: ", len(actor.user.users_to_pick_up), "\n", file=f)
        #             print("\n", file=f)
        for actor in run.actors:
            run_number_actors_dict[actor.service] += 1
            run_number_actors_dict["total"] += 1
            if(actor.service == "bus"):
                #if its serving users
                if(len(actor.user.users_to_pick_up) > 0):
                    run_emissions_dict[actor.service] += actor.emissions
                    run_emissions_dict["total"] += actor.emissions
            else:
                run_emissions_dict[actor.service] += actor.emissions
                run_emissions_dict["total"] += actor.emissions

            #have to add the actual users
            # run_number_users_dict[actor.service] += 1
            if(actor.service == "bus"):
                run_number_users_dict[actor.service] += len(actor.user.users_to_pick_up)
                run_number_users_dict["total"] += len(actor.user.users_to_pick_up)
            else:
                run_number_users_dict[actor.service] = run_number_users_dict[actor.service] + len(actor.user.users_to_pick_up) + 1
                run_number_users_dict["total"] = run_number_users_dict["total"] + len(actor.user.users_to_pick_up) + 1

            actor_transp_subsidy = 0.0
            if(actor.service == "bike"):
                continue
            if(actor.service !=  "bus"):
                actor_transp_subsidy += actor.calculate_transporte_subsidy(actor.user.house_node)
            for rider in actor.user.users_to_pick_up:
                actor_transp_subsidy += actor.calculate_transporte_subsidy(actor.rider_traveled_dist(rider.house_node))
            
            run_transport_subsidy_dict[actor.service] += actor_transp_subsidy
            run_transport_subsidy_dict["total"] += actor_transp_subsidy

        emissions_dict["car"].append(run_emissions_dict["car"])
        emissions_dict["bus"].append(run_emissions_dict["bus"])
        emissions_dict["sharedCar"].append(run_emissions_dict["sharedCar"])
        emissions_dict["bike"].append(run_emissions_dict["bike"])
        emissions_dict["total"].append(run_emissions_dict["total"])

        if(run_emissions_dict["total"] > max_emissions):
            max_emissions = run_emissions_dict["total"]

        tax_list.append(calculate_carbon_tax(run_emissions_dict["total"]))

        if(tax_list[-1] > max_carbon_tax):
            max_carbon_tax = tax_list[-1]

        number_users_dict["car"].append(run_number_users_dict["car"])
        number_users_dict["bus"].append(run_number_users_dict["bus"])
        number_users_dict["sharedCar"].append(run_number_users_dict["sharedCar"])
        number_users_dict["bike"].append(run_number_users_dict["bike"])
        number_users_dict["total"].append(run_number_users_dict["total"])

        number_actors_dict["car"].append(run_number_actors_dict["car"])
        number_actors_dict["bus"].append(run_number_actors_dict["bus"])
        number_actors_dict["sharedCar"].append(
            run_number_actors_dict["sharedCar"])
        number_actors_dict["bike"].append(run_number_actors_dict["bike"])
        number_actors_dict["total"].append(run_number_actors_dict["total"])

        transport_subsidy_dict["car"].append(run_transport_subsidy_dict["car"])
        transport_subsidy_dict["bus"].append(run_transport_subsidy_dict["bus"])
        transport_subsidy_dict["sharedCar"].append(
            run_transport_subsidy_dict["sharedCar"])
        transport_subsidy_dict["total"].append(
            run_transport_subsidy_dict["total"])

        if(run_transport_subsidy_dict["total"] > max_transport_subsidy):
            max_transport_subsidy = run_transport_subsidy_dict["total"]
        # print("sou o number users dict")
        # print(number_users_dict)

    # print("emission dict")
    # print(emissions_dict)
    # print("number user dict")
    # print(number_users_dict)

    #Calculate Emissions Tax - total and per run
    average_total_value_tax = sum(tax_list)/len(tax_list)

    print("vou fazer display das plots \n")

    if display_plots:
        plot_accumulated_actor_graph(actors_flow_acc, len(all_s))
        # plot_accumulated_edges_graphs(results['edges_occupation'], len(all_s))
        plot_emissions_development(emissions_dict)
        plot_number_users_development(number_users_dict)
    plt.waitforbuttonpress(0)  

    total_cost_list = []
    for i in range(0,len(all_s)):
        total_cost_list.append(tax_list[i] + transport_subsidy_dict["total"][i])
        i += 1

    max_combined_cost = max(total_cost_list)

    #get average travel times for all runs in the simulation
    average_ttt_all_runs = []
    for stats in all_s:
        run_ttt = 0
        for a in stats.actors:
            run_ttt += a.total_travel_time
        average_ttt_all_runs.append(run_ttt/len(stats.actors))

    # for key in dictionary.keys():
    #     print(key,file=f)
    #     run_res = dicitonary[key]
    #     for mode in fun_res:
    #         print(mode)
    #         print() 
   # print(users_lost)
    # print(users_lost.keys())
    # print(users_lost.values())
    # print(users_lost[0])

    # print("number actors dict")
    # print(number_actors_dict)
    # print("number users dict")
    # print(number_users_dict)

      #bus and shared car actors from the last 100 runs
    bus_last_100_actors = number_actors_dict["bus"][-last_runs:]
    shared_car_last_100_actors = number_actors_dict["sharedCar"][-last_runs:]

    #bus and shared car users from the last 100 runs
    bus_last_100_users = number_users_dict["bus"][-last_runs:]
    shared_car_last_100_users = number_users_dict["sharedCar"][-last_runs:]

    # print("sum bus users")
    # print(sum(bus_last_100_users))
    # print("sum bus actors")
    # print(sum(bus_last_100_actors))
    # print("sum shared car users")
    # print(sum(shared_car_last_100_users))
    # print("sum shared car actors")
    # print(sum(shared_car_last_100_actors))

    if(sum(bus_last_100_actors) == 0 or sum(shared_car_last_100_actors) == 0):
        if(sum(bus_last_100_actors) == 0 and sum(shared_car_last_100_actors) == 0):
            average_bus_occupancy = 0
            average_sharedCar_occupancy = 0
        elif(sum(bus_last_100_actors) == 0):
            average_bus_occupancy = 0
            average_sharedCar_occupancy = sum(
                shared_car_last_100_users)/sum(shared_car_last_100_actors)
        elif(sum(shared_car_last_100_actors) == 0):
            average_bus_occupancy = sum(
                bus_last_100_users)/sum(bus_last_100_actors)
            average_sharedCar_occupancy = 0
    else:
        average_bus_occupancy = sum(bus_last_100_users)/sum(bus_last_100_actors)
        average_sharedCar_occupancy = sum(
            shared_car_last_100_users)/sum(shared_car_last_100_actors)





    with open("{}_results.txt".format(run_name), 'a+') as f:
        print("Number users lost per run: \n", file=f)
        # print(users_lost, file=f)
        write_dict_file(users_lost,f)
        # print("\n", file=f)
        print("Min users lost: {} in run {} \n".format(min_val,min_run), file=f)
        print("Number of actors per mode: \n", file=f)
        write_dict_file(number_actors_dict,f)
        # print(number_actors_dict, file=f)
        # print("\n", file=f)
        print("Number of users per mode: \n", file=f)
        # print(number_users_dict, file=f)
        write_dict_file(number_users_dict, f)
        # print("\n", file=f)
        print("Average Bus Ocuppancy (last 100 runs): {} \n".format(average_bus_occupancy), file=f)
        print("Average Shared Ride Ocuppancy (last 100 runs): {} \n".format(average_sharedCar_occupancy), file=f)
        print("Emissions: \n", file=f)
        write_dict_file(emissions_dict, f)
        # print(emissions_dict, file=f)
        # print("\n", file=f)
        print("Carbon Tax: \n", file=f)
        print(tax_list, file=f)
        print("\n",file=f)
        print("Average Carbon Tax: \n", file=f)
        print(average_total_value_tax, file=f)
        print("\n", file=f)
        print("Transport subsidy: \n", file=f)
        write_dict_file(transport_subsidy_dict, f)
        # print(transport_subsidy_dict, file=f)
        # print("\n",file=f)
        print("Combined Cost (Carbon Tax + Transport Subsidy): \n", file=f)
        print(total_cost_list, file=f)
        print("\n", file=f)
        print("Max Emissions: \n", file=f)
        print(max_emissions, file=f)
        print("\n", file=f)
        print("Max Carbon Tax: \n", file=f)
        print(max_carbon_tax, file=f)
        print("\n", file=f)
        print("Max Transport Subsidy: \n", file=f)
        print(max_transport_subsidy, file=f)
        print("\n", file=f)
        print("Max Combined Cost (Carbon Tax + Transport Subsidy): \n", file=f)
        print(max_combined_cost, file=f)
        print("\n", file=f)
        print("Average and STD Total Travel Time: {} - {} \n".format(results["time_atis_no"][0], results["time_atis_no"][1]), file=f)
        print("Average Total Travel Time By Run: \n", file=f)
        print(average_ttt_all_runs, file=f)
        print("\n", file=f)
        print("Total Travel Time by mode (last 100 runs):", file=f)
        write_dict_file(time_per_mode_last_runs,f)

    return results

def write_dict_file(dictionary, f):
    for key in dictionary.keys():
        print(key,file=f)
        print(dictionary[key], file=f)
        print("\n",file=f)

def calculate_carbon_tax(emissions: float):
    return (emissions / 1000000) * CARBON_TAX

def read_json_file(file: str):
    f = open(file, "r")
    content = f.read()
    return json.loads(content)

def get_user_current_state(user: User):
    personality = user.personality
    return [user.start_time, personality.willingness_to_pay, personality.willingness_to_wait, personality.awareness, int(personality.has_private)]


def write_user_info(actors: List[Actor], run: int, file:str):
    run_header = ["Run"]
    run_row = [run]
    fields = ["Course", "Grade", "Cluster", "Willingness to pay", "Willingness to wait", "Awareness", "Comfort preference",
              "has private", "has bike","Friendliness", "Suscetible", "Transport", "Urban", "Willing", "Distance from Destination", "House Node", "Car Capacity", "Users he picked up","Transportation"]

    # writing to csv file
    
    with open(file, 'a+', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(run_header)
        csvwriter.writerow(run_row)
        # writing the fields
        csvwriter.writerow(fields)

        for act in actors:
            if(act.service != "bus"):
                us = act.user
                personality = us.personality
                info = [us.course, us.grade, us.cluster, personality.willingness_to_pay, personality.willingness_to_wait, personality.awareness, personality.comfort_preference,
                    personality.has_private, us.has_bike,personality.friendliness, personality.suscetible, personality.transport, personality.urban, personality.willing,
                        us.distance_from_destination, us.house_node, us.capacity, len(us.users_to_pick_up), us.provider.service]
                csvwriter.writerow(info)
            for rider in act.user.users_to_pick_up:
                personality = rider.personality
                info = [rider.course, rider.grade, rider.cluster, personality.willingness_to_pay, personality.willingness_to_wait, personality.awareness, personality.comfort_preference,
                        personality.has_private, rider.has_bike, personality.friendliness, personality.suscetible, personality.transport, personality.urban, personality.willing,
                        rider.distance_from_destination, rider.house_node, rider.capacity, len(rider.users_to_pick_up), rider.provider.service]
                csvwriter.writerow(info)
      


def main(args):
    ep_rewards = [0]

    #  Stats settings
    AGGREGATE_STATS_EVERY = 50  # episodes
    MIN_REWARD=1
    MODEL_NAME = 'Maas_simulator'

    if args.traffic_peaks is None:
        # Needed since "action=append" doesn't overwrite "default=X"
        args.traffic_peaks = [(8, 3), (18, 3)]

    print_args(args)

    input_config = read_json_file(args.json_file)

    providers = [Personal(),Friends(),STCP(), Bicycle()]
    # providers = [Personal(), STCP(), Bicycle()]
    # providers = [Personal()]
    # providers = [Personal(), STCP()]
    # providers = [ Friends()]
    # providers = [STCP()]
    # providers = [Bicycle()]
    sim = Simulator(config=args,
                    input_config = input_config,
                    actor_constructor=partial(
                        actor_constructor),
                    providers=providers,
                    stats_constructor=stats_constructor,
                    traffic_distribution=MultimodalDistribution(*args.traffic_peaks)
                    )
    n_inputs = 10
    n_output = len(providers)
    agent = DQNAgent(n_inputs, n_output)
    # gather stats from all runs
    all_stats = []

    last_runs = 100
    last_episodes = args.n_runs - last_runs

    time_per_mode_last_runs = {
        "Personal":0,
        "Friends":0,
        "STCP":0,
        "Bicycle":0
    }

    for episode in trange(args.n_runs, leave=False):
        # print(" episode")
        # print(episode)
        # Update tensorboard step every episode
        agent.tensorboard.step = episode
        if(episode == 0):
            sim.first_run = True
        else:
            sim.first_run = False
        #runs the simulation
        final_users = sim.run(agent)
        # final_users = sim.run_descriptive()
        
        # print("final users ", len(final_users))

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        # step = 1
           # Transform new continous state to new discrete state and count reward
        for user_info in final_users:
            episode_reward += user_info["utility"]
            if(episode >= last_episodes):
                time_per_mode_last_runs[user_info["commute_output"]
                                        .mean_transportation] += user_info["commute_output"].total_time



        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        # print("tenho o ep reqwrd")

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(
                ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=agent.epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    'models/{}__{}.model'.format(
                        MODEL_NAME, int(time.time())
                    ))

        # print("sai do if")
        # current_state, action, reward, new_current_state, done
        agent.update_epsilon()
        # print("update epsilon")
        
        # a = copy.deepcopy(sim.actors)
        copy_actors = []
        for actor in sim.actors:
            new_actor = actor.my_copy()
            copy_actors.append(new_actor)
        # print("deep copy")
        # sim.stats.add_actors(a) 
        sim.stats.add_actors(copy_actors)
        # print("adicionei actors")
        all_stats.append(sim.stats)
        # print("vou para o proximo run")

    #Add distance info to results file
    write_user_distance_interval_info(sim)
    json_object = average_all_results(
        all_stats, args.plots, sim.users_lost, time_per_mode_last_runs)
    json_object['graph'] = nx.readwrite.jit_data(sim.graph.graph)

    json.dump(json_object, open(args.save_path, "w+"))

    # print("all stats")
    # for i in all_stats:
    #     print("hi")
    #     print(i.actors)
    #     for ac in i.actors:
    #         print(ac.provider)

    # statistics_print(sim)

    save_user_file = run_name + "_users_info.csv"

    runn = 0
    for run in all_stats:
        write_user_info(run.actors, runn, save_user_file)
        runn += 1
    
def write_user_distance_interval_info(sim: Simulator):
    with open("{}_results.txt".format(run_name), 'w+') as f:
        print("Number of Users per distance interval: \n", file=f)
        write_dict_file(sim.distance_dict, f)

if __name__ == '__main__':
    main(parse_args())
