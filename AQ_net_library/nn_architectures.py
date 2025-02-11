import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np

#tf.config.optimizer.set_experimental_options({"xla": False})


# ==========================================================================================
# MISCELLANEOUS (noise generators and attention classes)
# ==========================================================================================

class RandomGaussianNoise(layers.Layer):
    def __init__(self, noise_level=0.2, noisy_proportion = 0.05, addNoiseInEval = False, **kwargs):
        super(RandomGaussianNoise, self).__init__(**kwargs)
        self.noise_level      =  noise_level
        self.noisy_proportion = noisy_proportion
        self.addNoiseInEval   = addNoiseInEval

    def get_config(self):
        config = super(RandomGaussianNoise, self).get_config().copy()
        config.update({"noise_level": self.noise_level,
                       "noisy_proportion": self.noisy_proportion})
        return config 

    def call(self, inputs,training=None):
        
        # if inference skip noise
        if not training and not self.addNoiseInEval:
            return inputs

        mask = tf.math.less(tf.random.uniform(tf.shape(inputs),minval = 0,maxval = 1), self.noisy_proportion)
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.noise_level, dtype=tf.float32)
        noisy_inputs = tf.where(mask, inputs + noise, inputs)
        
        return noisy_inputs 

    
class RandomLinearInterpolationNoise(layers.Layer):
    def __init__(self, Interp_probability=0.1, Max_inter_len=3*2*60, **kwargs):
        super(RandomLinearInterpolationNoise, self).__init__(**kwargs)
        self.Interp_probability = Interp_probability
        self.Max_inter_len = Max_inter_len

    def get_config(self):
        config = super(RandomLinearInterpolationNoise, self).get_config().copy()
        config.update({"Interp_probability": self.Interp_probability,
                       "Max_inter_len": self.Max_inter_len})
        return config 
 
    def call(self, inputs, training=None):
        # if inference skip noise
        if not training:
            return inputs    
        # if training add noise
        else:
            # Decide whether to add noise or not in the shape of a contiguous block
            random_val = tf.random.stateless_uniform(shape=[], seed=[0, 0], minval=0., maxval=1.)
            add_noise = tf.less(random_val, self.Interp_probability)
            
            def no_noise():
                return inputs
            
            def add_noise_fn():
                # Extracting the shape of the input tensor
                shape        = tf.shape(inputs)
                batch_size   = shape[0]
                num_points   = shape[1]
                num_channels = shape[2]
 
                # Generating the length of the block to interpolate
                inter_len   = tf.random.stateless_uniform([], seed=[0, 1], minval=1, maxval=self.Max_inter_len, dtype=tf.int32)
                # Generating the starting point of the block to interpolate
                MaxVal = num_points - inter_len
                start_point   = tf.random.stateless_uniform([], seed=[0, 2], minval=1, maxval=MaxVal, dtype=tf.int32)
                # Generating the ending point of the block to interpolate
                end_point = start_point + inter_len
 
                # Generating the mask for the block to interpolate
                mask = tf.concat([tf.zeros((batch_size, start_point, num_channels), dtype=tf.float32),
                                  tf.ones((batch_size, inter_len, num_channels), dtype=tf.float32),
                                  tf.zeros((batch_size, num_points - end_point, num_channels), dtype=tf.float32)], axis=1)
 
                # Linear interpolation between the start and stop values of the blocks
                ending_segment   = inputs[:, end_point, :]
                starting_segment = inputs[:, start_point, :]
 
                inter_len_f = tf.cast(inter_len, tf.float32)
                range_vals = tf.range(inter_len, dtype=tf.float32) / (inter_len_f - 1)
                range_vals = tf.reshape(range_vals, (1, inter_len, 1))
                linearly_spaced_values_t = starting_segment[:, None, :] + range_vals * (ending_segment[:, None, :] - starting_segment[:, None, :])
                
                linearly_spaced_values = tf.concat([tf.zeros((batch_size, start_point, num_channels), dtype=tf.float32),
                                                    linearly_spaced_values_t,
                                                    tf.zeros((batch_size, num_points - end_point, num_channels), dtype=tf.float32)], axis=1)
 
                outputs = inputs * (1 - mask) + mask * linearly_spaced_values
                return outputs
            
            return tf.cond(add_noise, add_noise_fn, no_noise)

class AttentionBlock(layers.Layer):

    def __init__(self, F_int, alpha_leaky_relu=0.1):
        """
        Initialize the AttentionBlock layer.

        Parameters:
        - F_int: Number of intermediate channels (internal).
        - alpha_leaky_relu: Slope for the LeakyReLU activation.
        """
        super(AttentionBlock, self).__init__()
        self.F_int = F_int
        self.alpha_leaky_relu = alpha_leaky_relu

        # Define the layers used in the block
        self.conv_g = layers.Conv1D(F_int, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.bn_g = layers.BatchNormalization()

        self.conv_x = layers.Conv1D(F_int, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.bn_x = layers.BatchNormalization()

        self.add = layers.Add()
        self.leaky_relu = layers.LeakyReLU(alpha=alpha_leaky_relu)

        self.conv_psi = layers.Conv1D(1, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.bn_psi = layers.BatchNormalization()

    def get_config(self):
        config = super(AttentionBlock, self).get_config().copy()
        config.update({"F_int": self.F_int,
                       "alpha_leaky_relu": self.alpha_leaky_relu,
                       "conv_g": self.conv_g,
                       "bn_g": self.bn_g,
                       "conv_x": self.conv_x,
                       "bn_x": self.bn_x,
                       "add": self.add,
                       "leaky_relu": self.leaky_relu,
                       "conv_psi": self.conv_psi,
                       "bn_psi": self.bn_psi})
        return config 

    def call(self, g, x):
        """
        Forward pass of the AttentionBlock.

        Parameters:
        - inputs: Tuple of (g, x), where
            g -- Gating signal input tensor.
            x -- Input feature map tensor.

        Returns:
        - Tensor after applying the attention mechanism.
        """
        # Gating signal transformation
        W_g = self.conv_g(g)
        W_g = self.bn_g(W_g)
        # Input feature map transformation
        W_x = self.conv_x(x)
        W_x = self.bn_x(W_x)
        # Combine and activate
        psi = layers.Add()([W_g, W_x])
        psi = self.leaky_relu(psi)
        # Generate attention weights
        psi = self.conv_psi(psi)
        psi = self.bn_psi(psi)
        psi = tf.keras.activations.sigmoid(psi)
        # Apply attention weights
        output = layers.Multiply()([x, psi])
        return output

    
# ==========================================================================================
# ARCHITECTURE: R2U-NET
# ==========================================================================================

class R2AttU_Net:

    """
    Classe WIP che serve a creare la rete Unet con attention layer e recurrent e recursive layers
    """
    
    def __init__(self, block_length, depth, n_filters=2, alpha_leaky_relu=0.1, kernel_size=13, seed=42, add_noise=False, addNoiseInEval = False):

        self.model = self.CreateAttentionUnetR2AttU_Net(block_length, depth=depth, n_filters=n_filters, alpha_leaky_relu=alpha_leaky_relu, kernel_size=kernel_size, seed=seed, add_noise=add_noise, addNoiseInEval = addNoiseInEval)

    def CreateAttentionUnetR2AttU_Net(self, block_length, depth, n_filters, kernel_size, alpha_leaky_relu, seed, add_noise, addNoiseInEval):
        
        # Kernel initializer
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed)
        # I make a list of convolutional layers to later retrieve for skip connections
        skip_layers = list()
        # INPUT LAYER
        input_layer = layers.Input((block_length, 1), name="Input")
        # Initializing the U-Net
        x = input_layer
        n = 0
        # Add noise to the signal
        if add_noise == True:
            x = RandomGaussianNoise(noise_level=0.1, addNoiseInEval = addNoiseInEval)(x)
            #x = RandomLinearInterpolationNoise(Interp_probability=0.1, Max_inter_len=3*2*60)(x)
        # The first block is different
        x = self.RRCNNBlock(x, n_filters*2**n, kernel_size, initializer, alpha_leaky_relu = alpha_leaky_relu) 
        skip_layers.append(x)
        # ENCODER LAYERS
        while n < depth:
            n += 1
            #ENCODER BLOCK
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = self.RRCNNBlock(x, n_filters*2**n, kernel_size, initializer, alpha_leaky_relu)
            if n < depth: 
                skip_layers.append(x)         
        # DECODER LAYERS
        while n > 0:
            #DECODER BLOCK
            x1 = layers.UpSampling1D(size=2)(x)
            x2 = AttentionBlock(n_filters*2**n, alpha_leaky_relu)(x1, skip_layers[n-1]) #def __init__(self, F_int, alpha_leaky_relu=0.1):
            x3 = layers.concatenate([x1, x2])
            x4 = self.RRCNNBlock(x3, n_filters*2**n, kernel_size, initializer, alpha_leaky_relu)
            x  = x4
            n -= 1
        # Output Layer
        output_layer = layers.Conv1D(filters=2, kernel_size=1, activation="softmax", kernel_initializer=initializer, name="Output")(x)
        # Assembling the model
        model = Model(inputs=[input_layer], outputs=[output_layer])
        return model
    def ConvStep(self, x, filters, kernel_size, alpha_leaky_relu, initializer, padding="same"):
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=None, kernel_initializer=initializer, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=alpha_leaky_relu)(x)
        return x
    def RecurrentBlock(self, x, filters, kernel_size, alpha_leaky_relu, initializer):
        x1 = self.ConvStep(x, filters, kernel_size, alpha_leaky_relu, initializer)
        x2 = layers.Add()([x1, x])
        x3 = self.ConvStep(x2, filters, kernel_size, alpha_leaky_relu, initializer)
        return x3
    def RRCNNBlock(self, x, filters, kernel_size, initializer, alpha_leaky_relu = 0.1):
        x = layers.Conv1D(filters=filters, kernel_size=1, activation=None, padding="same")(x)
        x1 = self.RecurrentBlock(x, filters, kernel_size, alpha_leaky_relu, initializer)
        x2 = self.RecurrentBlock(x1, filters, kernel_size, alpha_leaky_relu, initializer)
        x3 = layers.Add()([x, x2])
        return x3
    



 # ===================================================================================
# ATTENTION U NET
# ===================================================================================

class AttentionBlock2(layers.Layer):
    def __init__(self, in_channels, inter_channels=None, sub_sample_factor=2):
        """
        Initialize the Attention Block for 1D signals.
       
        Parameters:
        - in_channels: Number of channels in the input signal (x).
        - inter_channels: Number of intermediate channels.
        - sub_sample_factor: Factor to downsample the input signal.
        """
        super(AttentionBlock2, self).__init__()
       
        self.sub_sample_factor = sub_sample_factor
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2
       
        # Ensure the intermediate channels are at least 1
        if self.inter_channels == 0:
            self.inter_channels = 1
       
        # W_g: Transformation of the gating signal (g)
        self.W_g = layers.Conv1D(self.inter_channels, kernel_size=1, strides=1, padding="same", use_bias=True)
       
        # W_x: Transformation of the input signal (x)
        #self.W_x = layers.Conv1D(self.inter_channels, kernel_size=1, strides=self.sub_sample_factor, padding='valid', use_bias=False)
        self.W_x = layers.Conv1D(self.inter_channels, kernel_size=1, strides=self.sub_sample_factor, padding="same", use_bias=False)

        # psi: Compatibility function to produce attention mask (alpha)
        self.psi = layers.Conv1D(1, kernel_size=1, strides=1, padding="same", use_bias=True)
       
        # UpSampling layer to resize 1D signals
        self.upsample = layers.UpSampling1D(size=self.sub_sample_factor)


    def call(self, x, g):
        """
        Forward pass of the Attention Block for 1D signals.

        Parameters:
        - x: Input signal tensor of shape (batch, length, channels).
        - g: Gating signal tensor of shape (batch, reduced_length, gating_channels).

        Returns:
        - W_y: Output tensor after attention is applied.
        - alpha: Attention mask tensor (final attention weights).
        """

        # Transform the gating signal (g) using W_g
        phi_g = self.W_g(g)  # Shape: (batch, reduced_length, inter_channels)
       
        # Transform the input signal (x) using W_x
        theta_x = self.W_x(x)  # Shape: (batch, length/sub_sample_factor, inter_channels)
       
        # Element-wise sum and ReLU activation
        f = tf.nn.relu(theta_x + phi_g)
       
        # Generate attention mask (alpha)
        alpha = tf.nn.sigmoid(self.psi(f))  # Shape: (batch, length/sub_sample_factor, 1)
       
        # Upsample the attention mask to match the original input size
        alpha_resampled = self.upsample(alpha)
       
        # Apply attention mask to the input signal (x)
        y = alpha_resampled * x
       
        return y, alpha_resampled

class AttentionU_net:

    def __init__(self,block_length, depth, n_filters=8, factor=2, alpha_leaky_relu=0.1, kernel_size=15, seed=42, add_noise=False, addNoiseInEval = False):
        self.block_length     = block_length
        self.depth            = depth
        self.n_filters        = n_filters
        self.factor           = factor
        self.alpha_leaky_relu = alpha_leaky_relu
        self.kernel_size      = kernel_size
        self.seed             = seed
        self.add_noise        = add_noise
        self.addNoiseInEval   = addNoiseInEval

    def build_model(self):
        return self.Build_Attention_UNET(self.block_length,
                                         self.depth,
                                         self.n_filters,
                                         self.factor,  
                                         self.kernel_size,
                                         self.alpha_leaky_relu,
                                         self.seed,
                                         self.add_noise,
                                         self. addNoiseInEval)

    def Build_Attention_UNET(self, block_length, depth, n_filters, factor, kernel_size, alpha_leaky_relu, seed, add_noise, addNoiseInEval):

        # Convolutional step helper function
        def ConvStep(x, n_filters, kernel_size, alpha_leaky_relu, initializer, padding="same"):
            x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=None, kernel_initializer=initializer, padding=padding)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=alpha_leaky_relu)(x)
            return x
       
        # Kernel initializer
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed)

        # I make a list of layers to later retrieve for skip connections
        skip_layers = list()

        # INPUT LAYER
        input_layer = layers.Input((block_length, 1), name="Input")
        x = input_layer
        n = 0 # Initializing variable to keep track of depth of U-NET

        # Add noise to the signal
        if add_noise == True:
            x = RandomGaussianNoise(noise_level=0.1, addNoiseInEval = addNoiseInEval)(x)
            #x = RandomLinearInterpolationNoise(Interp_probability=0.1, Max_inter_len=3*2*60)(x)
           
        # ENCODER LAYERS
        while n < depth -1:
            x = ConvStep(x, n_filters*(2**n), kernel_size, alpha_leaky_relu, initializer, padding="same")
            x = ConvStep(x, n_filters*(2**n), kernel_size, alpha_leaky_relu, initializer, padding="same")
            if n < depth:
                skip_layers.append(x)
                x = tf.keras.layers.MaxPooling1D(pool_size=factor, name=str(n+1)+"_Encoder_MaxPool")(x)    
            n += 1

        # Subtract one since it was not needed in the end
        n -= 1

        # Final conv step
        x = ConvStep(x, n_filters*(2**n), kernel_size, alpha_leaky_relu, initializer, padding="same")
        x = ConvStep(x, n_filters*(2**n), kernel_size, alpha_leaky_relu, initializer, padding="same")

        # DECODER LAYERS
        while n > 0:

            # Get the skip connection from the encoder path
            skip_connection = skip_layers[n]

            # Attention gate
            attention_output, attention_map = AttentionBlock2(in_channels=skip_connection.shape[-1], sub_sample_factor=factor)(skip_connection, x)
            x = tf.keras.layers.Conv1DTranspose(filters=n_filters*(2**n), kernel_size=kernel_size, strides=factor, padding='same')(x)
            x = tf.keras.layers.concatenate([x, attention_output])
            # Apply the convolutional block after concatenation
            x = ConvStep(x, n_filters*(2**n), kernel_size, alpha_leaky_relu, initializer, padding="same")
            x = ConvStep(x, n_filters*(2**n), kernel_size, alpha_leaky_relu, initializer, padding="same")
            # Move to the next depth level
            n -= 1
       
        # Output Layer
        output_layer = layers.Conv1DTranspose(filters=2, kernel_size=kernel_size, strides=factor, padding='same', activation='softmax')(x)
       

        # Assembling the model
        model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
        return model







# ==========================================================================================
# ARCHITECTURE: U-NET
# ==========================================================================================

class U_net():

    def __init__(self, block_length, depth=4, n_filters=8, factor=2, alpha_leaky_relu=0.1, kernel_size=15, seed=42, add_noise=False, addNoiseInEval = False):
        # DESCRIPTION
        # - block_length: the length of the input chunk of the timeseries
        # - depth: the depth of the neural network
        # - n_filters: the number of filters to use for convolution
        # - factor: the downsample and upsample factor
        # - alpha_leaky_relu: the alpha of the leaky relu activation function
        # - kernel_size: the kernel size for the convolution
        # - seed: the random seed for reproducibility
        self.model = self.CreateUnet(block_length=block_length, depth=depth, n_filters=n_filters, factor=factor, alpha_leaky_relu=alpha_leaky_relu, kernel_size=kernel_size, seed=seed, add_noise=add_noise, addNoiseInEval = addNoiseInEval)


    def CreateUnet(self, block_length, depth, n_filters, factor, alpha_leaky_relu, kernel_size, seed, add_noise, addNoiseInEval):
        # Kernel initializer
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed)
        # I make a list of convolutional layers to later retrieve for skip connections
        conv_layers = list()
        # INPUT LAYER
        input_layer = layers.Input((block_length, 1), name="Input")
        # Initializing the U-Net
        x = input_layer
        n = 0
        # Add noise to the signal
        if add_noise == True:
            x = RandomGaussianNoise(noise_level=0.1, addNoiseInEval = addNoiseInEval)(x)
            #x = RandomLinearInterpolationNoise(Interp_probability=0.1, Max_inter_len=3*2*60)(x)
        # ENCODER LAYERS
        while n < depth:
            # Block 1
            x = layers.Conv1D(filters=n_filters*2**(n), kernel_size=kernel_size, activation=None, kernel_initializer=initializer, padding="same", name=str(n+1)+"_Encoder_Conv_1")(x)
            x = layers.LeakyReLU(alpha=alpha_leaky_relu, name=str(n+1)+"_Encoder_Activation_1")(x)
            x = layers.BatchNormalization(name=str(n+1)+"_Encoder_BatchNorm_1")(x)
            # Block 2
            x = layers.Conv1D(filters=n_filters*2**(n), kernel_size=kernel_size, activation=None, kernel_initializer=initializer, padding="same", name=str(n+1)+"_Encoder_Conv_2")(x)
            x = layers.LeakyReLU(alpha=alpha_leaky_relu, name=str(n+1)+"_Encoder_Activation_2")(x)
            x = layers.BatchNormalization(name=str(n+1)+"_Encoder_BatchNorm_2")(x)
            conv_layers.append(x)  
            # Maxpool (but not at the last depth)
            if n < depth-1:
                x = layers.MaxPooling1D(pool_size=factor, name=str(n+1+1)+"_Encoder_MaxPool")(x)     
            n += 1
        # Decrease by 1 as the last adding was not needed
        n -= 1
        # DECODER LAYERS
        while n > 0:
            # Conv1DTranspose
            x = layers.Conv1DTranspose(filters=n_filters*2**(n-1), kernel_size=factor, strides=factor, padding="same", name=str(n+1)+"_Decoder_UpConv")(x)
            # Concatenate
            x = layers.concatenate([x, conv_layers[n-1]], name=str(n+1)+"_Decoder_Skip")
            # Block 1
            x = layers.Conv1D(n_filters*2**(n-1), kernel_size, activation=None, kernel_initializer=initializer, padding="same", name=str(n)+"_Decoder_Conv_1")(x)
            x = layers.LeakyReLU(alpha=alpha_leaky_relu, name=str(n)+"_Decoder_Activation_1")(x)
            x = layers.BatchNormalization(name=str(n)+"_Decoder_BatchNorm_1")(x)
            # Block 2
            x = layers.Conv1D(n_filters*2**(n-1), kernel_size, activation=None, kernel_initializer=initializer, padding="same", name=str(n)+"_Decoder_Conv_2")(x)
            x = layers.LeakyReLU(alpha=alpha_leaky_relu, name=str(n)+"_Decoder_Activation_2")(x)
            x = layers.BatchNormalization(name=str(n)+"_Decoder_BatchNorm_2")(x)
            n -= 1
        # Output Layer
        output_layer = layers.Conv1D(filters=2, kernel_size=1, activation="softmax", kernel_initializer=initializer, name="Output")(x)
        # Assembling the model
        model = Model(inputs=[input_layer], outputs=[output_layer])
        return model