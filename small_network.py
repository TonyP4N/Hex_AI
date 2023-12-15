import numpy as np
from inputFormat import *
import tensorflow as tf
from tensorflow.keras import layers


class Network(tf.keras.Model):
	def __init__(self, output_dim=(11, 11)):
		super().__init__()
		# Convolutional layers
		self.conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')
		self.conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')
		self.conv3 = layers.Conv2D(6, (5, 5), activation='relu', padding='same')
		self.conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
		self.conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
		
		# Flattening and Dense layers
		self.flatten = layers.Flatten()
		self.dense1 = layers.Dense(1024, activation='relu')
		
		# Output layer
		self.dense2 = layers.Dense(output_dim[0] * output_dim[1])

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)

		# Reshaping the output to the desired shape
		return tf.softmax(tf.reshape(x, [-1, *(11, 11)]))

class PolicyNetwork(tf.keras.Model):
	def __init__(self, num_channels, input_size, boardsize, batch_size=1):
		super().__init__()
		self.layer0 = HexConvLayer(20, 12)
		self.layer1 = HexConvLayer(16, 16)
		self.layer2 = HexConvLayer(12, 20)
		self.layer3 = HexConvLayer(8, 24)
		self.layer4 = HexConvLayer(4, 28)
		self.layer5 = HexConvLayer(0, 32)
		self.layer6 = tf.keras.layers.Dense(boardsize * boardsize)

	def call(self, inputs):
		x = self.layer0(inputs)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = tf.reshape(x, [self.batch_size, -1])  # Flatten
		x = self.layer6(x)


		x = tf.reshape(x, [self.batch_size, -1])
		x = self.layer6(x)

		# Assume 'inputs' has shape [batch_size, num_channels, board_height, board_width]
		# Define the 'not_played' mask
		white, black = 0, 1  # indices for white and black channels
		not_played = tf.logical_and(
			tf.equal(inputs[:, white, :, :], 0),
			tf.equal(inputs[:, black, :, :], 0)
		)

		# Flatten 'not_played' to match the output shape
		not_played_flat = tf.reshape(not_played, [self.batch_size, -1])

		# Apply mask to the network's output
		masked_output = tf.where(not_played_flat, x, tf.fill(tf.shape(x), -np.inf))

		# Apply softmax to the masked output
		playable_output = tf.nn.softmax(masked_output)

		return playable_output