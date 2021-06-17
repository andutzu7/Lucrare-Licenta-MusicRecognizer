import matplotlib
import numpy as np
import copy
from matplotlib import pyplot as plt

step = []
acc = []
loss = []
with open('trainingingfinal','r') as f:
    file = f.readlines()
    lines = [line.rstrip('\n') for line in file]
    for i in range(len(lines)):
        out = lines[i].split(',')
        step.append(int(out[0]))
        acc.append(float(out[1]))
        loss.append(float(out[2]))

plt.plot(loss)
plt.show()