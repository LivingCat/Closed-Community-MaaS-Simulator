# How to run
python src/main.py -r 100 -js ex.json

tensorboard --logdir logs/2x256-1579515056

# Resources
* [Tutorial Deep Q-Learning](https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/)
* [Video tutorial Deep Q-Learning](https://www.youtube.com/watch?v=qfovbG84EBg&t=4s)

# Runs
* muda numero de pessoas para ver traffic:
  * 100 check
  * 300 check
  * 500 check
* muda os custos
  * menos para autocarro check
  * mais para carro(portagem) +1 check
* mudar o tempo
  * autocarro check
* mudar conforto
  * autocarro check
* mais pessoas a partilhar


3000 episodes
* run 1
  * 100 pessoas
  * tudo igual
* run 2
  * 300 pessoas
  * tudo igual
* run 3
  * 500 pessoas
  * tudo igual

* run 4
  * 1 custo autocarro
* run 5
  * custo carro + 2 custo shared + 2/3
* run 6
  * mudar tempo do autocarro + ou - dependendo se pessoas gostam do autocarro
* run 7
  * mudar conforto do autocarro + ou - dependendo se pessoas gostam do autocarro


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
* Utility factors- how to account for sociability? For now we are going to only have the awareness of the impact of the chosen service
* Actor occupation chart uses only last few runs

# Done
* Make different actors - bus,car,shared car
* Make edges only let certain actors traverse them
* Create routes for actors according to the edges specifications
* Parser for graph

# Things to have until 15/12
* [X] Make different actors - bus,car,shared car
* [X] Make edges only let certain actors traverse them
* [X] Create routes for actors according to the edges specifications
* [X] Added Users
* [ ] Utility function
* [X] Reinforcement Learning
* [ ] Users can only choose private car if they have a car
* [ ] Utility function can't be divided by 0


# Might have
* Gif showing the occupation of the edges
* Gif showing people learning

# Probs not
* Each service has one color

# 09-12-2019

* In the process of changing actors_flow to be a dictionary for all services (actors)

# 10-12-2019

* Charts now show statistics for different actors - actor chart and edge chart

# 11-12-2019

* Graph is created based on a json file (no longer hardcoded)

# 12-12-2019

* Added Users
* Users choose (randomly) a transportation method
* Users have the "result" of the commute (time, cost, awareness)

# 13-12-2019

* Added Reinforcement learning

# 14-12-2019

* Tensorboard
* Tried different personality types - need to work on that

# 14-04-2020

* Finaly made a commit with all the changes as of yet
* Users have more information from clusters, factors, courses and grades
* Writing user info to a file at the end of the main (need to think about a new place)

# 18-02-2020

* Users now have salary and budget
* Fixed willingness to pay so that it now has to do with the salary of the user