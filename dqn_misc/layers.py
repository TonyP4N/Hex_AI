import tensorflow as tf
import numpy as np

class HexConvLayer(tf.keras.layers.Layer):
    def __init__(self, num_D5_filters, num_D3_filters, **kwargs):
        super(HexConvLayer, self).__init__(**kwargs)
        self.num_D5_filters = num_D5_filters
        self.num_D3_filters = num_D3_filters

    def build(self, input_shape):
        W3_bound = np.sqrt(6. / (7*(input_shape[-1] + self.num_D3_filters)))
        W5_bound = np.sqrt(6. / (19*(input_shape[-1] + self.num_D5_filters)))

        self.W3_values = self.add_weight(shape=(self.num_D3_filters, input_shape[-1], 7),
                                         initializer=tf.random_uniform_initializer(minval=-W3_bound, maxval=W3_bound),
                                         trainable=True)

        self.W5_values = self.add_weight(shape=(self.num_D5_filters, input_shape[-1], 19),
                                         initializer=tf.random_uniform_initializer(minval=-W5_bound, maxval=W5_bound),
                                         trainable=True)

        self.b = self.add_weight(shape=(self.num_D5_filters+self.num_D3_filters,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        # Reshape or manipulate W3_values and W5_values to fit hexagonal structure
        # This is a conceptual example and would need to be adapted to your specific hexagonal filter format
        hex_W3 = self.reshape_to_hexagonal(self.W3_values, diameter=3)
        hex_W5 = self.reshape_to_hexagonal(self.W5_values, diameter=5)

        # Apply convolution with hexagonal weights
        conv_out3 = tf.nn.conv2d(inputs, hex_W3, strides=[1, 1, 1, 1], padding="VALID")
        conv_out5 = tf.nn.conv2d(inputs, hex_W5, strides=[1, 1, 1, 1], padding="VALID")

        # Concatenate outputs and apply activation
        full_out = tf.concat([conv_out5, conv_out3], axis=-1)
        activated_out = tf.nn.relu(full_out + self.b)

        # Return the final output
        return activated_out

    def reshape_to_hexagonal(self, weights, diameter):
        if diameter == 3:
        # Creating a mask for a hexagon within a 3x3 kernel
            mask = tf.constant([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=weights.dtype)
        elif diameter == 5:
            # Creating a mask for a hexagon within a 5x5 kernel
            mask = tf.constant([[0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=weights.dtype)
        else:
            raise ValueError("Unsupported diameter for hexagonal kernel")

        # Reshape the weights to a 4D tensor if they are not already
        reshaped_weights = tf.reshape(weights, [weights.shape[0], 3, 3, weights.shape[-1]])

        # Apply the mask to simulate a hexagonal structure
        hexagonal_weights = reshaped_weights * mask

        return hexagonal_weights

    def compute_output_shape(self, input_shape):
        # Define how to compute output shape from input shape
        return input_shape  # Modify based on the layer's operation

    def get_config(self):
        config = super(HexConvLayer, self).get_config()
        config.update({'num_D5_filters': self.num_D5_filters,
                       'num_D3_filters': self.num_D3_filters})
        return config
    
