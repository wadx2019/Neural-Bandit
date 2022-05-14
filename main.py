from env import *
from neuralucb import *
from neuralts import *
import matplotlib.pyplot as plt
import torch
from tqdm import trange


env = IRIS()


linucb = LinUCB(env.dim, env.arm, beta=0.2, lamb=1)
LinUCBregret = [0]


for i in range(10000):
    context, reward = env.step()
    arm = linucb.take_action(context)
    LinUCBregret += [LinUCBregret[-1] + 1 - reward[arm]]
    linucb.update(context, arm, reward[arm])


plt.plot(LinUCBregret, label='LinUCB')
print('linucb:', LinUCBregret[-1])


lints = LinTS(env.dim, env.arm, beta=0.1, lamb=1)
LinTSregret = [0]


for i in range(10000):
    context, reward = env.step()
    arm = lints.take_action(context)
    LinTSregret += [LinTSregret[-1] + 1 - reward[arm]]
    lints.update(context, arm, reward[arm])


plt.plot(LinTSregret, label='LinTS')
print('lints:', LinTSregret[-1])

neuralucb = NeuralUCB(env.dim, env.arm, beta=1, lamb=1)
NeuralUCBregret = [0]


for i in trange(10000):
    context, reward = env.step()
    arm = neuralucb.take_action(context)
    NeuralUCBregret += [NeuralUCBregret[-1] + 1 - reward[arm]]
    neuralucb.update(context, arm, reward[arm])


plt.plot(NeuralUCBregret, label='NeuralUCB')
print('neuralucb:', NeuralUCBregret[-1])

neuralts = NeuralTS(env.dim, env.arm, beta=1, lamb=1)
NeuralTSregret = [0]


for i in trange(10000):
    context, reward = env.step()
    arm = neuralts.take_action(context)
    NeuralTSregret += [NeuralTSregret[-1] + 1 - reward[arm]]
    neuralts.update(context, arm, reward[arm])


plt.plot(NeuralTSregret, label='NeuralTS')
print('neuralts:', NeuralTSregret[-1])

plt.grid()
plt.legend(['LinUCB',  'LinTS', 'NeuralUCB', 'NeuralTS'])
plt.show()