"""
Graph representing a road network.
Graph topology should allow for dynamic run-time changes (e.g. accidents
and other phenomena that restrict or even block a given edge).
"""
from typing import List, Tuple
from utils import congestion_time_estimate

import networkx as nx


class RoadGraph:

    graph: nx.DiGraph
    nstart: int
    nend: int

    def __init__(self,input_config):
        self.create_graph(input_config)
        # self.mass_hardcoded_graph()

    def __print_edge_volumes(self):
        """Pretty print of the edges current volumes. Useful for debug purposes"""
        print("Volumes:")
        for e in self.graph.edges:
            print("\t(%i, %i) -> %i" %
                (e[0], e[1], self.graph.edges[e[0], e[1]]['volume']))

    def add_vehicle(self, edge: (int, int)):
        """Add a vehicle to a given edge"""
        self.graph.edges[edge[0], edge[1]]['volume'] += 1

    def remove_vehicle(self, edge: (int, int)):
        """Remove a vehicle from a given edge"""
        self.graph.edges[edge[0], edge[1]]['volume'] -= 1

    def get_edge_data(self, edge: Tuple[int, int]) -> dict:
        """Get edge related data."""
        return self.graph.edges[edge[0], edge[1]]

    def get_possible_routes(self, src_node: int, dest_node: int, actor: str):
        """Get all possible routes from the src_node to the destiny_node"""
        possible_edges = []

        for edge in self.graph.edges:
            allowed_transports = (self.get_edge_data(edge)['allowed_transports'])
            if(actor in allowed_transports):
                possible_edges.append(edge)
        new_graph = nx.DiGraph(possible_edges)
        a = nx.all_simple_paths(new_graph, src_node, dest_node)
        # i=0
        # for path in a:
        #     print(path)
        #     print(i)
        #     i +=1
        a = list(a)
        return a

    def check_has_route(self, start_node:int, actor:str):
        possible_edges = []
        for edge in self.graph.edges:
            allowed_transports = (self.get_edge_data(edge)[
                                    'allowed_transports'])
            if(actor in allowed_transports):
                possible_edges.append(edge)
        new_graph = nx.DiGraph(possible_edges)

        #if both start and end node are in this new graph than there exists a path between them
        if(new_graph.has_node(start_node) and new_graph.has_node(self.nend)):
            return True
        else:
            return False

    def get_all_routes(self, start_node:int, actor: str) -> List[List[int]]:
        # results in [[0, 1, 3], [0, 2, 1, 3], [0, 2, 3]]
        return self.get_possible_routes(start_node, self.nend, actor)
        # this below doesn't work bc it forces to go through all nodes
        # return nx.all_topological_sorts(self.graph)

    def get_optimal_route_travel_time(self, route: List[int]) -> float:
        """Gets the estimated optimal travel time it takes to transverse a given route"""
        edges = list(zip(route, route[1:]))

        estimates = [self.graph.edges[e[0], e[1]]['free_flow_travel_time']
                     for e in edges]

        return sum(estimates)

    def get_edge_travel_time(self, edge: Tuple[int, int], volume: int, service:str) -> float:
        """Get the time it takes to transverse the edge, considering a given volume"""
        edge_data = self.get_edge_data(edge)
        return congestion_time_estimate(edge_data['free_flow_travel_time'],
                                        edge_data['capacity'],
                                        volume, service)

    def get_edge_real_travel_time(self, edge: Tuple[int, int], service:str) -> float:
        """Get the real actual time it takes to transverse the edge (congestion included)"""
        return self.get_edge_travel_time(edge, self.get_edge_data(edge)['volume'],service)

    def create_graph(self, input_config):
        """Creates a graph from from input file configuration"""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(0, int(input_config["graph"]["num_nodes"])))
        self.nstart = int(input_config["graph"]["start_node"])
        self.nend = int(input_config["graph"]["end_node"])
        edges_list = input_config["graph"]["edges_list"]
        for key in edges_list.keys():
            self.graph.add_edges_from(
                [(int(key.split(',')[0]), int(key.split(',')[1]))], 
                color=edges_list[key]["color"],
                allowed_transports=edges_list[key]["allowed_transports"],
                volume=edges_list[key]["volume"],
                free_flow_travel_time=edges_list[key]["free_flow_travel_time"],
                capacity=edges_list[key]["capacity"]
                )
