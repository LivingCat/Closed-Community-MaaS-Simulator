# How to run
python src/main.py -r 100


# To Do
* Add people that choose a mean of transport
* Write stats to a file
* Think about others possible parsers, maybe for population
* Think about attributes the population will have - relationships 
* Utility function - paper
* Reinforcement learning
* Stats about occupancy of edges, type of actors, number people chose each service, utility
* Bus event 10 in 10 minutes? 
* Incentives - think about how Cat can introduce them later in an easy breazy way

# Done
* Make different actors - bus,car,shared car
* Make edges only let certain actors traverse them
* Create routes for actors according to the edges specifications
* Parser for graph

# Things to have until 15/12
* Make different actors - bus,car,shared car
* Make edges only let certain actors traverse them
* Create routes for actors according to the edges specifications

# Might have
* Gif showing the occupation of the edges

# Probs not
* Each service has one color

# 09-12-2019

* In the process of changing actors_flow to be a dictionary for all services (actors)

# 10-12-2019

* Charts now show statistics for different actors - actor chart and edge chart

# 11-12-2019

* Graph is created based on a json file (no longer hardcoded)