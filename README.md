# Neural-Bandit Algorithms (NeuralUCB and NeuralTS)

A collection of the pytorch implementation of neural bandit algorithm includes neuralUCB [Neural Contextual Bandits with UCB-based Exploration](https://arxiv.org/pdf/1911.04462.pdf) and neural Thompson sampling [NEURAL THOMPSON SAMPLING](https://arxiv.org/pdf/2010.00827.pdf).

This project contrasts the performance of neuralUCB and neuralTS with that of LinUCB and LinTS over the IRIS dataset.

In the consideration of the lack of an modularized and open-source neural bandit algorithm collection, this project designed a general neural bandit framework to make it convenient for users to apply neural bandit algorithms in their own project. 

The simulation result is shown below.

![result](https://github.com/wadx2019/Neural-bandit/blob/main/result.png)

To obtain this performance, you can simply run main.py.
