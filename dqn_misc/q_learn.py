import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time
import pickle
from small_network import *

def save():
	print("saving network...")
	if args.save:
		save_name = args.save
	else:
		save_name = "Q_network"
	network.save(save_name)
	if args.data:
		f = open(args.data+"/replay_mem.save", 'w')
		pickle.dump(mem, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		f = open(args.data+"/costs.save","w")
		pickle.dump(costs, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()
		f = open(args.data+"/values.save","w")
		pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def show_plots():
	print(costs)
	plt.figure(0)
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('episode')
	plt.draw()
	plt.pause(0.001)
	plt.figure(1)
	plt.plot(values)
	plt.ylabel('value')
	plt.xlabel('episode')
	plt.draw()
	plt.pause(0.001)

def epsilon_greedy_policy(state, evaluator):
	rand = np.random.random()
	played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
	if(rand>epsilon_q):
		state = tf.convert_to_tensor(state, dtype=tf.float32)
		scores = evaluator(tf.expand_dims(state, axis=0))
		#set value of played cells impossibly low so they are never picked
		played_indices = np.where(played)[0]//11
		played_indices2 = np.where(played)[0]%11
		# print(played_indices)
		# print(played_indices2)
		# Assuming 'scores' is your TensorFlow tensor
		scores = scores.numpy()  # Convert to NumPy array
		# print(scores.shape)
		# Assuming 'played_indices' is an array of indices where you want to set values to -2
		for x in range(len(played_indices)):
			scores[0][played_indices[x]][played_indices2[x]] = -2

		# Convert back to TensorFlow tensor
		#np.set_printoptions(precision=3, linewidth=100)
		#print scores.max()
		return scores.argmax(), scores.max()
	#choose random open cell
	return np.random.choice(np.arange(boardsize*boardsize)[np.logical_not(played)]), 0


def Q_update(network, optimizer, mem, batch_size):
	states1, actions, rewards, states2 = mem.sample_batch(batch_size)

	# Convert data to TensorFlow tensors
	states1 = tf.convert_to_tensor(states1, dtype=tf.float32)
	actions = tf.convert_to_tensor(actions, dtype=tf.int32)
	rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
	states2 = tf.convert_to_tensor(states2, dtype=tf.float32)
	# Forward pass for the next states
	future_scores = network(states2, training=False)

	# Custom logic for played states
	played = np.logical_or(states2[:, white, padding:boardsize+padding, padding:boardsize+padding],
							states2[:, black, padding:boardsize+padding, padding:boardsize+padding])
	future_scores = tf.where(played, -2 * tf.ones_like(future_scores), future_scores)

	# Assuming 'rewards' is a 1D tensor of shape [batch_size]
	batch_size = tf.shape(rewards)[0]

	# Create a tensor of ones with the same shape as the result of the 'else' clause
	then_tensor = tf.fill([batch_size], 1.0)

	# Now use these in the tf.where function
	targets = tf.where(rewards == 1, then_tensor, -tf.reduce_max(-tf.reduce_max(future_scores, axis=1),-1))

	# Train step
	with tf.GradientTape() as tape:
		predictions = network(states1, training=True)
		actions_taken = tf.one_hot(actions, depth=predictions.shape[-1])
		actions_taken_expanded = tf.expand_dims(actions_taken, -1)
		q_values = tf.reduce_sum(tf.reduce_sum(predictions * actions_taken_expanded, axis=1),axis=1)
		# q_values = tf.reduce_sum(predictions * actions_taken, axis=0)
		loss = tf.reduce_mean(tf.square(q_values - targets))

	gradients = tape.gradient(loss, network.trainable_variables)
	optimizer.apply_gradients(zip(gradients, network.trainable_variables))

	return loss

def action_to_cell(action):
	cell = np.unravel_index(action, (boardsize,boardsize))
	return(cell[0]+padding, cell[1]+padding)

def flip_action(action):
	return boardsize*boardsize-1-action

class replay_memory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.size = 0
		self.index = 0
		self.full = False
		self.state1_memory = np.zeros(np.concatenate(([capacity], input_shape)), dtype=bool)
		self.action_memory = np.zeros(capacity, dtype=np.uint8)
		self.reward_memory = np.zeros(capacity, dtype=bool)
		self.state2_memory = np.zeros(np.concatenate(([capacity], input_shape)), dtype=bool)

	def add_entry(self, state1, action, reward, state2):
		self.state1_memory[self.index, :, :] = state1
		self.state2_memory[self.index, :, :] = state2
		self.action_memory[self.index] = action
		self.reward_memory[self.index] = reward
		self.index += 1
		if(self.index>=self.capacity):
			self.full = True
			self.index = 0
		if not self.full:
			self.size += 1

	def sample_batch(self, size):
		batch = np.random.choice(np.arange(0,self.size), size=size)
		states1 = self.state1_memory[batch]
		states2 = self.state2_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		return (states1, actions, rewards, states2)


parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

#save network every x minutes during training
save_time = 5
#save snapshot of network to unique file every x minutes during training
snapshot_time = 10

print("loading starting positions... ")
datafile = open("../train_data/scoredPositionsFull.npz", 'rb')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

replay_capacity = 100000

if args.data:
	if not os.path.exists(args.data):
		os.makedirs(args.data)
		mem = replay_memory(replay_capacity)
		costs = []
		values = []
	else:
		if os.path.exists(args.data+"/replay_mem.save"):
			print("loading replay memory...")
			f = open(args.data+"/replay_mem.save")
			mem = pickle.load(f)
			f.close
		else:
			#replay memory from which updates are drawn
			mem = replay_memory(replay_capacity)
		if os.path.exists(args.data+"/costs.save"):
			f = open(args.data+"/costs.save")
			costs = pickle.load(f)
			f.close
		else:
			costs = []
		if os.path.exists(args.data+"/values.save"):
			f = open(args.data+"/values.save")
			values = pickle.load(f)
			f.close
		else:
			values = []
else:
	#replay memory from which updates are drawn
	mem = replay_memory(replay_capacity)
	costs = []
	values = []

if args.load:
    print("Loading model...")
    network = tf.keras.models.load_model(args.load)
else:
    print("Building model...")
    boardsize = 11  # Replace with your board size
    network = Network()  # Adapt parameters


numEpisodes = 20
batch_size = 64
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)


print("Running episodes...")
epsilon_q = 0.1
last_save = time.process_time()
last_snapshot = time.process_time()
show_plots()
try:
	for i in range(numEpisodes):
		cost = 0
		num_step = 0
		value_sum = 0
		#randomly choose who is to move from each position to increase variability in dataset
		move_parity = np.random.choice([True,False])
		#randomly choose starting position from database
		index = np.random.randint(numPositions)
		#randomly flip states to capture symmetry
		if(np.random.choice([True,False])):
			gameW = np.copy(positions[index])
		else:
			gameW = flip_game(positions[index])
		gameB = mirror_game(gameW)
		t = time.process_time()
		while(winner(gameW)==None):
			action, value = epsilon_greedy_policy(gameW if move_parity else gameB, network)
			value_sum+=abs(value)
			state1 = np.copy(gameW if move_parity else gameB)
			move_cell = action_to_cell(action)
			play_cell(gameW, move_cell if move_parity else cell_m(move_cell), white if move_parity else black)
			play_cell(gameB, cell_m(move_cell) if move_parity else move_cell, black if move_parity else white)
			if(not winner(gameW)==None):
				#only the player who just moved can win, so if anyone wins the reward is 1
				#for the current player
				reward = 1
			else:
				reward = 0
			#randomly flip states to capture symmetry
			if(np.random.choice([True,False])):
				state2 = np.copy(gameB if move_parity else gameW)
			else:
				state2 = flip_game(gameB if move_parity else gameW)
			move_parity = not move_parity
			mem.add_entry(state1, action, reward, state2)
			if(mem.size > batch_size):
				cost += Q_update(network,optimizer,mem,batch_size).numpy()

				#print state_string(gameW)
			num_step += 1
			if(time.process_time()-last_save > 60*save_time):
				save()
				show_plots()
				last_save = time.process_time()
			if(time.process_time()-last_snapshot > 60*snapshot_time):
				save()
				last_snapshot = time.process_time()
		run_time = time.process_time() - t
		print("Episode", i, "complete, cost: ", 0 if num_step == 0 else cost/num_step, " Time per move: ", 0 if num_step == 0 else run_time/num_step, "Average value magnitude: ", 0 if num_step == 0 else value_sum/num_step)
		costs.append(0 if num_step == 0 else cost/num_step)
		values.append(0 if num_step == 0 else value_sum/num_step)

except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save()
	exit(1)

save()
