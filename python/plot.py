import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

fname = "aht-non-act"
x = np.array(range(10, 100, 10))
y = []
std = []
fin = open(fname + ".txt")
for line in fin.readlines():
    ax, astd = line.split(',')
    y.append(int(ax))
    std.append(int(astd))
ax = plt.errorbar(x, y, std, linestyle='-', marker='o')

fname = "aht-act"
x = np.array(range(10, 100, 10))
y = []
std = []
fin = open(fname + ".txt")
for line in fin.readlines():
    ax, astd = line.split(',')
    y.append(int(ax))
    std.append(int(astd))
ax = plt.errorbar(x, y, std, linestyle='-', marker='x')

fname = "aht-staged"
x = np.array(range(10, 100, 10))
y = []
std = []
fin = open(fname + ".txt")
for line in fin.readlines():
    ax, astd = line.split(',')
    y.append(int(ax))
    std.append(int(astd))
erbc = plt.errorbar(x, y, std, linestyle='-', marker='+')

plt.legend(["Non-Adaptive User Sampling", "Adaptive User Sampling", "Two Stage Ranking"])
fmt = matplotlib.pyplot.ScalarFormatter()
erbc[0].axes.yaxis.set_major_formatter(fmt)
plt.xlabel("Number of items to rank")
plt.ylabel("Sample Complexity")
plt.title("$\gamma_A = 0.5, \gamma_B = 5.0$")
plt.savefig('nonacac1.pdf')
