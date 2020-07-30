#https://stackoverflow.com/questions/17658512/how-to-pipe-input-to-python-line-by-line-from-linux-program
import sys
import numpy as np

times = []
for line in sys.stdin:
    times.append(line)

ntimes = np.array(times, dtype=float)
print("Items: ", ntimes.shape)
print("Mean: ", np.mean(ntimes))
print("Max: ", np.amax(ntimes))
print("Min: ", np.amin(ntimes))