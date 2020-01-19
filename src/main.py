"""
Main project source file.
"""
from typing import List, Tuple
from ipdb import set_trace
from actor import Actor
from data_plotting import plot_accumulated_actor_graph, plot_accumulated_edges_graphs, plot_emissions_development, plot_number_users_development
from simulator import Simulator
from provider import Provider, Personal, Friends, STCP
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
run_name = "cheaper_bus_100"

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
    possible_routes = graph.get_all_routes(user.mean_transportation)
    routes_times = [graph.get_optimal_route_travel_time(r)
                    for r in possible_routes]
    routes_probs = softmax_travel_times(routes_times)
    idx = np.random.choice(len(possible_routes), p=routes_probs)

    return Actor(possible_routes[idx], user, user.provider)



def stats_constructor(graph: RoadGraph):
    # print("Created STATS")
    return SimStats(graph)


def statistics_print(sim: Simulator):
    """Print of simulation statistics regarding ATIS and non ATIS users"""
    print()
    atis_no = []
    for a in sim.actors:
        # if a.atis is not None:
        #     atis_yes.append(a.total_travel_time)
        # else:
        atis_no.append(a.total_travel_time)

    # print("ATIS YES: mean: %f || std: %f" %
    #       (np.mean(atis_yes), np.std(atis_yes)))
    print("ATIS NO: mean: %f || std: %f" % (np.mean(atis_no), np.std(atis_no)))


def average_all_results(all_s: List[SimStats], display_plots: bool):
    """Gather information regarding all runs and its metrics"""

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

    results['actors_atis_natis'] = actors_flow_acc

    # the above but for every edge
    inner_default_dict = lambda: defaultdict(lambda: [])
    results['edges_occupation'] = defaultdict(inner_default_dict)

    for s in all_s:
        edges = s.edges_flow_atis
        for key in edges.keys():
            for service in edges[key]:
                results['edges_occupation'][str(
                    key)][service].append(edges[key][service])
    

    # pretty(results['edges_atis_natis'])
    # print(results['edges_atis_natis'])

    for e_key in results['edges_occupation'].keys():
        edge_flow = defaultdict(lambda: [(0.0, 0)])
        # pretty(results['edges_occupation'][e_key])
        for actor_key in results['edges_occupation'][e_key]:
            for tuple_list in results['edges_occupation'][e_key][actor_key]:
                tuple_list = tuple_list[1:]
                for tuple in tuple_list:
                    edge_flow[actor_key].append(tuple)

        for key in edge_flow.keys():
            edge_flow[key] = sorted(edge_flow[key], key=lambda t: t[0])

        edge_flow_acc = defaultdict(lambda: [(0.0, 0)])
        for actor_key in edge_flow.keys():
            for edge_tuple in edge_flow[actor_key]:
                edge_flow_acc[actor_key].append([edge_tuple[0],
                                    edge_tuple[1] + edge_flow_acc[actor_key][-1][1]])


        for actor_key in edge_flow_acc.keys():
            edge_flow_acc[actor_key] = edge_flow_acc[actor_key][1:]

        # print("acc")
        # print(edge_flow_acc)
        results['edges_occupation'][e_key] = edge_flow_acc
    emissions_dict={
        "car":[],
        "bus":[],
        "sharedCar":[],
        "total":[]
    }

    number_users_dict = {
        "car": [],
        "bus": [],
        "sharedCar": []
    }

    for run in all_s:
        run_emissions_dict={
            "car": 0,
            "bus" :0,
            "sharedCar": 0,
            "total": 0
        }
        run_number_users_dict = {
            "car": 0,
            "bus" :0,
            "sharedCar": 0
        }
        for actor in run.actors:
            run_emissions_dict[actor.service] += actor.emissions
            run_number_users_dict[actor.service] += 1
            run_emissions_dict["total"] += actor.emissions

        emissions_dict["car"].append(run_emissions_dict["car"])
        emissions_dict["bus"].append(run_emissions_dict["bus"])
        emissions_dict["sharedCar"].append(run_emissions_dict["sharedCar"])
        emissions_dict["total"].append(run_emissions_dict["total"])

        number_users_dict["car"].append(run_number_users_dict["car"])
        number_users_dict["bus"].append(run_number_users_dict["bus"])
        number_users_dict["sharedCar"].append(run_number_users_dict["sharedCar"])

    np.save("{}_emissions".format(run_name),np.array(emissions_dict))
    np.save("{}_number".format(run_name), np.array(number_users_dict))

    if display_plots:
        plot_accumulated_actor_graph(actors_flow_acc, len(all_s))
        plot_accumulated_edges_graphs(results['edges_occupation'], len(all_s))
        plot_emissions_development(emissions_dict)
        plot_number_users_development(number_users_dict)
    plt.waitforbuttonpress(0)  


    return results

def read_json_file(file: str):
    f = open(file, "r")
    content = f.read()
    return json.loads(content)

def get_user_current_state(user: User):
    personality = user.personality
    return [user.start_time, personality.willingness_to_pay, personality.willingness_to_wait, personality.awareness, int(personality.has_private)]

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

    providers = [Personal(),Friends(),STCP()]
    sim = Simulator(config=args,
                    input_config = input_config,
                    actor_constructor=partial(
                        actor_constructor),
                    providers=providers,
                    stats_constructor=stats_constructor,
                    traffic_distribution=MultimodalDistribution(*args.traffic_peaks)
                    )
    n_inputs = 6
    n_output = len(providers)
    agent = DQNAgent(n_inputs, n_output)
    # gather stats from all runs
    all_stats = []
    for episode in trange(args.n_runs, leave=False):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        final_users = sim.run(agent)

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1
           # Transform new continous state to new discrete state and count reward
        for user_info in final_users:
            episode_reward += user_info["utility"]


        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)


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
        # current_state, action, reward, new_current_state, done
        agent.update_epsilon()
        sim.stats.add_actors(sim.actors)
        all_stats.append(sim.stats)


    json_object = average_all_results(all_stats, args.plots)
    json_object['graph'] = nx.readwrite.jit_data(sim.graph.graph)

    json.dump(json_object, open(args.save_path, "w+"))

    statistics_print(sim)

if __name__ == '__main__':
    main(parse_args())
