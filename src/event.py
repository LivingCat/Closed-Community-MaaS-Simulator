"""
Events in the simulation process.
E.g. create_actor, travel_route.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
from user import User

import random

import simulator
import actor



class Event(ABC):
    """Class to encapsulate a simulation Event"""

    def __init__(self, at_time: float):
        super().__init__()
        self.at_time = at_time

    @abstractmethod
    def act(self, sim) -> List[Event]:
        pass

    def __lt__(self, other):
        return self.at_time < other.at_time

    def __eq__(self, other):
        return self.at_time == other.at_time

    def get_priorized(self) -> Tuple[float, Event]:
        return (self.at_time, self)

    def get_timestamp(self) -> float:
        return self.at_time


class CreateActorEvent(Event):
    """Event associated to the creation of an actor.
    It will trigger an edge start event for the route first edge."""

    def __init__(self, at_time: float, actor_constructor, user: User):
        super().__init__(at_time)
        self.actor_constructor = actor_constructor
        self.user = user

    def act(self, sim: simulator.Simulator):

        random_num = random.randint(0, 10)
        if(random_num < 3):
            a = self.actor_constructor(sim.graph, 'CarActor', self.user)
        elif(random_num < 9):
            a = self.actor_constructor(sim.graph, 'BusActor', self.user)
        else:
            a = self.actor_constructor(sim.graph, 'SharedCarActor', self.user)

        sim.actors.append(a)

        if sim.config.verbose:
            print("%f" % round(self.at_time, 5), " -- created actor %d" %
                  (a.actor_id))

        # updating general stats only
        sim.stats.add_actor(a,self.at_time)
        a.start_trip(self.at_time)

        return [EdgeStartEvent(self.at_time,
                               a,
                               a.get_next_travel_edge(self.at_time))]


class EdgeStartEvent(Event):
    """
    Represents point in time in which an Actor starts travelling along an Edge.
    """

    def __init__(self, at_time: float, a: actor.AbstractActor, edge: Tuple[int, int]):
        super().__init__(at_time)
        self.actor = a
        self.edge = edge

    def act(self, sim: simulator.Simulator):
        """
        Updates simulator's statistics (e.g. increase load/traffic on edge).
        """
        sim.stats.add_actor_edge(self.actor,
            self.at_time, self.edge)

        tt = sim.graph.get_edge_real_travel_time(self.edge)
        sim.graph.add_vehicle(self.edge)

        self.actor.add_time_for_edge(self.edge, tt)
        # sim.draw_graph()
        return [EdgeEndEvent(self.at_time + tt, self.actor, self.edge)]


class EdgeEndEvent(Event):
    """
    Represents point in time in which an Actor terminates travelling along an Edge.
    """

    def __init__(self, at_time: float, a: actor.AbstractActor, edge: Tuple[int, int]):
        super().__init__(at_time)
        self.actor = a
        self.edge = edge

    def act(self, sim: simulator.Simulator):
        """
        Updates simulator's statistics (e.g. decrease load/traffic on edge),
        and creates following EdgeStartEvent (if trip is not over).
        """
        sim.stats.remove_actor_edge(self.actor,
            self.at_time, self.edge)

        self.actor.travel(self.at_time, self.edge)
        sim.graph.remove_vehicle(self.edge)

        if not self.actor.reached_dest():
            # sim.draw_graph()
            # Time it starts next edge its equal to the time this event ended
            return [EdgeStartEvent(self.at_time, self.actor, self.actor.get_next_travel_edge(self.at_time))]

        if sim.config.verbose:
            self.actor.print_traveled_route()

        # updating general stats
        self.actor.update_total_tt()
        sim.stats.remove_actor(self.actor,self.at_time)
        return []


class AccidentEvent(Event):
    """Represents an unexpected negative event on the network (e.g. traffic accidents)"""

    def __init__(self, at_time: float, edge: Tuple[int, int], scale_factor: float):
        super().__init__(at_time)
        self.edge = edge                        # edge to target
        # how much to scale target by (e.g. edge's capacity)
        self.scale_factor = scale_factor

    def act(self, sim) -> List[Event]:
        sim.graph.graph.edges[self.edge]['capacity'] *= self.scale_factor
        return []
