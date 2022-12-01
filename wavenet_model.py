import tensorflow as tf
import numpy as np
import os

class WaveNet_model():

    def __init__(self, nb_layers:int, audio_length:int):
        
        input = tf.keras.Input(shape = (audio_length,1), name="WN Input")

        # 1st Causal convolution of the model, with 1 filter and taking to account the whole audio file: kernel_size=audio_length
        out = tf.keras.layers.Conv1D(filters=1, kernel_size=audio_length,padding='causal', name="Initial_Causal_Conv")(input)
        
        # -------- 1st residual bloc --------
        # delated causal convolutions

        # out = tf.keras.layers.Conv1D(filters=1, kernel_size=2, padding='causal', name='Delated_Causal_Conv_1',dilation_rate = 2**0)(out)
        # out = tf.keras.layers.Conv1D(filters=1, kernel_size=2, padding='causal', name='Delated_Causal_Conv_2',dilation_rate = 2**1)(out)
        # out = tf.keras.layers.Activation('sigmoid')(out)

        skip_connection = []
        for k in range(nb_layers):
            out, skipx = self.__residual_block(out)
            skip_connection.append(skipx)

        # Final layers before the output

        skip_out = tf.keras.layers.Add()(skip_connection)
        # skip_out = tf.keras.layers.Activation('relu')(skip_out)
        # skip_out = tf.keras.layers.Conv1D(filters=1,kernel_size=1, padding='same')(skip_out)
        # skip_out = tf.keras.layers.Activation('relu')(skip_out)
        # skip_out = tf.keras.layers.Conv1D(filters=1,kernel_size=1, padding='same')(skip_out)
        # skip_out = tf.keras.layers.Activation('softmax')(skip_out)
        

        self.model = tf.keras.models.Model(inputs = input, outputs = skip_out)




    def __residual_block(self,x):
        # x : output of the 
        # delated conv filter
        delated_conv_f0 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2, dilation_rate = 2**0)(x)
        delated_conv_f1 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**1)(delated_conv_f0)
        delated_conv_f2 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**2)(delated_conv_f1)
        delated_conv_f3 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**3)(delated_conv_f2)
        delated_conv_f4 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**4)(delated_conv_f3)
        delated_conv_f5 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**5)(delated_conv_f4)
        delated_conv_f6 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**6)(delated_conv_f5)
        delated_conv_f7 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**7)(delated_conv_f6)
        delated_conv_f8 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**8)(delated_conv_f7)
        delated_conv_f9 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**9)(delated_conv_f8)
        tanh_out = tf.keras.layers.Activation('tanh')(delated_conv_f9)

         # delated conv gate

        delated_conv_g0 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2, dilation_rate = 2**0)(x)
        delated_conv_g1 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**1)(delated_conv_g0)
        delated_conv_g2 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**2)(delated_conv_g1)
        delated_conv_g3 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**3)(delated_conv_g2)
        delated_conv_g4 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**4)(delated_conv_g3)
        delated_conv_g5 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**5)(delated_conv_g4)
        delated_conv_g6 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**6)(delated_conv_g5)
        delated_conv_g7 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**7)(delated_conv_g6)
        delated_conv_g8 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**8)(delated_conv_g7)
        delated_conv_g9 = tf.keras.layers.Conv1D(filters=1,padding = "causal", kernel_size = 2,dilation_rate = 2**9)(delated_conv_g8)
        sigmoid_out = tf.keras.layers.Activation('sigmoid')(delated_conv_g9)

        # merge x tanh & sigmoid
        multiply = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])
        conv11 = tf.keras.layers.Conv1D(filters=1,kernel_size=1, padding='same')(multiply)
        skipx = tf.keras.layers.Conv1D(filters=1,kernel_size=1, padding='same')(multiply)
        resx = tf.keras.layers.Add()([x, conv11])
        
        
        return resx, skipx