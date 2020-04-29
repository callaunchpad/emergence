import os, sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

FILE = sys.argv[1]
with open(FILE, 'r') as fp:
	data = fp.read().split("\n")
	data = [line.strip() for line in data if line.strip()]
	data = data[2:]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

arrX = []
arrY = []
idx = 0
for line in data:
	idx += 1
	line = [item.strip() for item in line.split(",")]
	rew = [float(x) for x in line[0][1:-1].strip().split(" ") if x]
	# print(line, rew)
	ep_len = int(line[1])
	win_one = rew[0] > 0
	win_two = rew[1] > 0
	arrX.append(idx)
	arrY.append((win_one, win_two, ep_len))

N = 100
plt.plot(arrX[0:-N+1], moving_average(np.array([y[2] for y in arrY]), n=N))
# plt.gcf().autofmt_xdate()
plt.show()
