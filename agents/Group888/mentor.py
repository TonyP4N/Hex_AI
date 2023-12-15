import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time
from small_network import *

parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
parser.add_argument("--data", "-d", type=str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

# Load Data
print("Loading data... ")
datafile = np.load("../train_data/scoredPositionsFull.npz")
positions = datafile['positions'].astype(np.float32)
scores = datafile['scores'].astype(np.float32)
print(positions.shape)
print(scores.shape)
n_train = scores.shape[0]

# Prepare Data for Training
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((positions, scores))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Initialize or Load Network
if args.load:
    print("Loading model...")
    network = tf.keras.models.load_model(args.load)
else:
    print("Building model...")
    boardsize = 11  # Replace with your board size
    network = Network()  # Adapt parameters

# Compile the Model
network.compile(optimizer='rmsprop', loss='mean_squared_error')

# Training Loop
num_epochs = 100
costs = []
print("Training model on mentor set...")
try:
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_cost = 0
        for batch, (p_batch, s_batch) in enumerate(dataset):
            t = time.time()
            cost = network.train_on_batch(p_batch, s_batch)
            epoch_cost += cost
            run_time = time.time() - t
            print("Cost: ", epoch_cost / (batch + 1), " Time per position: ", run_time / batch_size)
        costs.append(epoch_cost / (batch + 1))

        # Plotting
        plt.plot(costs)
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.draw()
        plt.pause(0.001)

        # Save the network
        if args.save:
            network.save(args.save)

except KeyboardInterrupt:
    # Save on interrupt
    if args.save:
        network.save(args.save)

# Final Save
if args.save:
    network.save(args.save)

print("Done training!")