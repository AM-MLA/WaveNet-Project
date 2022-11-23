import tensorflow as tf
import numpy as np
import os

class WaveNet_model():

    def __init__(self, nb_layers:int, nb_filters:int, audio_length:int):
        
        self.nb_filters= nb_filters

        #à compléter
        inputWN = tf.keras.Input(shape = (audio_length, 256), name="WN Input")

        out = tf.keras.layers.Conv1D(filters=nb_filters,kernel_size=2, padding="causal", name='initial_causal_convolution')(inputWN)

        skip_connections = []
        for k in range(nb_layers):
            out, skipx = self.__residual_block(out)
            skip_connections.append(skipx)

        skip_out = tf.keras.layers.Add()(skip_connections)
        skip_out = tf.keras.layers.Activation('relu')(skip_out)
        skip_out = tf.keras.layers.Conv1D(filters=256, kernel_size=1, padding = "same")(skip_out)
        skip_out = tf.keras.layers.Activation('relu')(skip_out)
        skip_out = tf.keras.layers.Conv1D(filters=256, kernel_size=1, padding = "same")(skip_out)


        outputWN =  tf.keras.layers.Activation('softmax', name="WN_Output")(skip_out)

        self.model = tf.keras.models.Model(inputs = inputWN, outputs = out)
        
        # return WNmodel


    def __delated_conv(self, input, nb_dilatation : int) -> tf.keras.layers.Conv1D :
        for i in range(nb_dilatation):
            delated_conv_f = tf.keras.layers.Conv1D(dilation_rate = 2**i) (delated_conv_f)


    def __residual_block(self,x):
        # x : output of the 
        # delated conv filter
        delated_conv_f0 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2, dilation_rate = 2**0)(x)
        delated_conv_f1 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**1)(delated_conv_f0)
        delated_conv_f2 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**2)(delated_conv_f1)
        delated_conv_f3 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**3)(delated_conv_f2)
        delated_conv_f4 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**4)(delated_conv_f3)
        delated_conv_f5 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**5)(delated_conv_f4)
        delated_conv_f6 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**6)(delated_conv_f5)
        delated_conv_f7 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**7)(delated_conv_f6)
        delated_conv_f8 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**8)(delated_conv_f7)
        delated_conv_f9 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**9)(delated_conv_f8)
        tanh_out = tf.keras.layers.Activation('tanh')(delated_conv_f9)

         # delated conv gate

        delated_conv_g0 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2, dilation_rate = 2**0)(x)
        delated_conv_g1 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**1)(delated_conv_g0)
        delated_conv_g2 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**2)(delated_conv_g1)
        delated_conv_g3 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**3)(delated_conv_g2)
        delated_conv_g4 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**4)(delated_conv_g3)
        delated_conv_g5 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**5)(delated_conv_g4)
        delated_conv_g6 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**6)(delated_conv_g5)
        delated_conv_g7 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**7)(delated_conv_g6)
        delated_conv_g8 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**8)(delated_conv_g7)
        delated_conv_g9 = tf.keras.layers.Conv1D(filters=self.nb_filters,padding = "causal", kernel_size = 2,dilation_rate = 2**9)(delated_conv_g8)
        sigmoid_out = tf.keras.layers.Activation('sigmoid')(delated_conv_g9)

        # merge x tanh & sigmoid
        multiply = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])

        # conv 1,1
        resx = tf.keras.layers.Conv1D(filters=self.nb_filters,kernel_size=1, padding='same')(multiply)
        skipx = tf.keras.layers.Conv1D(filters=self.nb_filters,kernel_size=1, padding='same')(multiply)
        resx = tf.keras.layers.Add()([x, resx])

        return resx, skipx
        
    
