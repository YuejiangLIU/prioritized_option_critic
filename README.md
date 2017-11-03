# Prioritized Option-Critic

## Motivation
* Longer options are preferred for efficient decision making
* Sooner and more certain rewards are likely received when actions are executed without option deliberation

## Formulation
![](images/formulation.png)

## Result
#### Learning process
![](images/durations.png) | ![](images/steps.png)
:-------------------------:|:-------------------------:
```
python transfer_priority.py --baseline --discount=0.9 --epsilon=0.01 --noptions=4 --lr_critic=0.5 --lr_intra=0.25 --lr_term=0.25 --nruns=100 --nepisodes=5000 --nsteps=1000 --priority=5
```

The goal is changed very 1000 episodes. --noptions=4 --nruns=100 --discount=0.9<sub>

Along with the learning process, the average duration of options keeps growing when the option taken previously is prioritized.

#### Learned option
η = 1 | ![](images/pterm-eta_1-opt_4-1.png) | ![](images/pterm-eta_1-opt_4-2.png) | ![](images/pterm-eta_1-opt_4-3.png) | ![](images/pterm-eta_1-opt_4-4.png)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
η = 5 | ![](images/pterm-eta_5-opt_4-1.png) | ![](images/pterm-eta_5-opt_4-2.png) | ![](images/pterm-eta_5-opt_4-3.png) | ![](images/pterm-eta_5-opt_4-4.png)
η = 20 | ![](images/pterm-eta_20-opt_4-1.png) | ![](images/pterm-eta_20-opt_4-2.png) | ![](images/pterm-eta_20-opt_4-3.png) | ![](images/pterm-eta_20-opt_4-4.png)
```
python transfer_priority.py --baseline --discount=0.9 --epsilon=0.01 --noptions=4 --lr_critic=0.5 --lr_intra=0.25 --lr_term=0.25 --nruns=1 --nepisodes=50000 --nsteps=1000 --priority=5
```

Options are learned in 50000 episodes, during which the goal is changed very 1000 episodes. The options learned from the original option-critic are chopped into primitive actions, whereas the options learned from the prioritized option-critic are more sustained.

## References
- [The Option-Critic Architecture on arXiv:1609.05140](https://arxiv.org/abs/1609.05140)
- [The Option-Critic Architecture on GitHub](https://github.com/jeanharb/option_critic/tree/master/)

## Dependencies
- Numpy
- Argparse
- matplotlib
- dill
- gym 0.7.2
