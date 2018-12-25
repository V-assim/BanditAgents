# Bandit algorithms.
For a Reinforcement Learning class, I worked on a few popular algorithms for the bandit problem. Among them were :

* Epsilon-greedy bandit
* BESA
* Softmax
* UCB1
* Thompson sampling
* KL-UCB

Bandits are implemented in `agent.py`

## How to use ?
For the purpose of the class, each agent was tested on a specific configuration : 1000 rounds for 2000 agents in parallel :
`python main.py --niter 1000 --batch 2000`
Use `python main.py -h` to know more.
