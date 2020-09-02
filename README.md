Starting code for simulator from [HERE](https://github.com/xfontes42/hermes-simulation)

# Simulator for Mobility-as-a Service (MaaS)
This project is a simulator written in Python used to simulate the use of different types of Mobility as a Service solutions (MaaS).
The project receives a JSON file characterizing an agent population and saves the agents belonging to the population  and proceedes to ......

The simulator outputs some graphs to show how the evolution of the utility for each of the ways of transport and the C02 emissions resulted from commuting of the agents throughtout the simulation run.

# Dependencies

- NetworkX: for modelling the graph/road network.
- Numpy: for data handling.
- Pandas: for data handling.
- Matplotlib: for data visualization.
- Tensorflow and Keras: for deep learning.
- Scipy: for probability distributions of the characteristics of the population

To install the dependencies I created an environment in anaconda. The file environment_file.yml has the information about the multiple dependencies.
```bash
conda env create -f environment_file.yml
```
To activate the environment:
```bash
conda activate myenv
```

# Usage

To run the project 
```bash
python main.py [-n N] [-r R] [-thr THRESH] [-tmax MAX_TIME]
                [-p TPEAK_MEAN TPEAK_STD] [-o SAVE_PATH] [-v] [-js POP_SETTINGS] [-save POP_FILE]
```

Required Arguments:

- -js,--json          json file that includes multiple settings related to the population and the graph

Optional Arguments:

  - -n N, --num_actors N  number of agents to generate per simulation
                        run
 - -r R, --runs R        number of times to run the simulation for
 - -save,--save        csv file that has the characteristics for a population of agents. Used to save and import the agents population

