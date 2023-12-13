import numpy as np
from resistance import score
from preprocess import *

# result = []
# with open("train_data/training_data.txt", 'r') as f:
# 	for line in f:
# 		line = line.strip()
# 		line = line.split(" ")
# 		if line[0] == line[1]:
# 			result.append(' '.join(line[1:]))
# 		else:
# 			result.append(' '.join(line))

# print(result)
# with open("train_data/training_data.dat", 'w') as f:
# 	for line in result:
# 		f.write(line + '\n')

positions = preprocess("train_data/training_data.dat")
print ("scoring positions...")
scores = np.empty((positions.shape[0],boardsize,boardsize))
num_positions = positions.shape[0]
output_interval = num_positions/100
for i in range(num_positions):
	if(i%output_interval == 0):
		print ("completion: ",i/output_interval)
	try:
		scores[i]=score(positions[i], 0)
	#if for some reason an uncaught singularity occurs just skip this position
	except np.linalg.linalg.LinAlgError:
		print ("singular position at ",str(i),": ", state_string(positions[i]))
		i-=1

print ("saving to file...")
savefile = open("train_data/scoredPositionsFull2.npz", 'wb')
np.savez(savefile, positions=positions, scores=scores)