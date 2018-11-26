# Utility Plotting Program for Hwk 4.
# Author Marcelo Siero
# CS220 Advanced Parallel Processing,
# Fall of 2018.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# plt.show()
cpu_elapsed_time = 14752.9639

my_cpu_elapsed_time = 1670

elapsed = np.array(
    [
        244, 124, 63, 34, 22, 14, 123, 62, 32, 18, 12, 13, 61, 31, 16, 9, 12, 11, 31, 16, 10, 10, 9, 9, 33, 17, 10, 8, 8, 9, 24, 13, 8, 7, 8, 9
    ])

blockSize = []
gridSize = []
totalThreads = []

byGridSize = dict()
byBlockSize = dict()
byTotalThreads= dict()

startBlockSize = 32
endBlockSize = 1024
startGridSize = 16
endGridSize = 512

gridSizeList = []
gSz = startGridSize
while (gSz<=endGridSize):
   gridSizeList.append(gSz)
   gSz *= 2

blockSizeList  = []
bSz = startBlockSize
while (bSz <= endBlockSize):
   blockSizeList.append(bSz)
   bSz *= 2

# Create a list of values for gridSize and blockSize corresponding
# to the elapsed times for plotting.
# This is a Python equivalent of this C code:
#  for (gSz=startGridSize; gSz<=endGridSize; gSz *= 2):
#     for (bSz = startBlockSize; bSz <= endBlockSize; bSz *= 2):
inx = 0
for gSz in gridSizeList:
   for bSz in blockSizeList:
      blockSize.append(bSz)
      gridSize.append(gSz)
      totalForThreads = bSz * gSz
      totalThreads.append(totalForThreads)
      # list distributes by blocksize corresponding to different threadSizes
      byGridSize.setdefault(bSz, []).append(elapsed[inx])
      # list distributes by threadSize corresponding to different blockSizes
      byBlockSize.setdefault(gSz, []).append(elapsed[inx])
      # list distributes by totalThreads
      byTotalThreads[totalForThreads] = elapsed[inx]
      inx += 1

# DO ALL THE PLOTS:
###############################################################
# Manage colors
# colors = cm.rainbow(np.linspace(0, 1, len(ys)))
# colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
#    Need to look up cm
# cs = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]
# cs = [colors[i//len(X)] for i in range(len(Ys)*len(X))] #could be done with numpy's repmat
###############################################################
## red dashes: r--, blue squares: b and green triangles
## Save the figure
totalThreadsSorted = sorted(byTotalThreads.keys())
timesByTotalThreads = [byTotalThreads[key] for key in totalThreadsSorted]
plt.figure(1)
plt.clf()    # clear figure
plt.title('Time vs Number of Threads')
plt.ylabel('Time in Msec.')
plt.xlabel('Total Threads')
plt.plot(totalThreadsSorted, timesByTotalThreads)
plt.savefig("time_vs_total_threads", bbox_inches="tight")
plt.close()

###############################################################
fig = plt.figure(2)
plt.title('Time versus blockDim (TPB)')
axes = fig.add_subplot(1,1,1)
# axes.set_xlim([1, len(gridSizeList[0])])
plt.ylabel('Time in Msec.')
plt.xlabel('blockDim (threads per block)')
# given a single gridSize show relation to blockSize (TPB).
for gSz in gridSizeList:
   plt.plot(blockSizeList, byBlockSize[gSz])
lines = axes.get_lines()
legend = plt.legend([lines[i] \
   for i in range(len(lines))], blockSizeList, loc=1, title='gridSize')
axes.add_artist(legend)
plt.savefig("time_vs_blockDim", bbox_inches="tight")
plt.close()

###############################################################
fig = plt.figure(3)
plt.title('Time versus gridDim (Blocks/Grid)')
axes = fig.add_subplot(1,1,1)
plt.xlabel('gridDim (blocks per thread)')
plt.ylabel('Time in Msec.')
for bSz in blockSizeList:
   plt.plot(gridSizeList, byGridSize[bSz])
lines = axes.get_lines()
legend = plt.legend([lines[i] \
   for i in range(len(lines))], blockSizeList, loc=1, title='blockSize')
axes.add_artist(legend)
plt.savefig("time_vs_gridDim", bbox_inches="tight")
plt.close()
