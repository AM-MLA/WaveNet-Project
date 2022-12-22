import tensorflow as tf


class WavenetModel:
    """This class creates the model.
        -   nb_residual: int    |   number of residual blocks to stack.
        -   nb_dilatation : int |   number of piles of 10 dilated convolution to be stacked.
        -   audio_length: int   |   size of the audio input.
    """

    def __init__(self, nb_residual, nb_dilatation, audio_length, **kwargs):
        input = tf.keras.Input(shape=(audio_length, 256,), name="WaveNet_Input")
        # the kernel size corresponds to the receptive field of the network,
        # corresponds to the dilatation factor of the dilated convolution: 2⁹⁺¹
        self.causal_conv = tf.keras.layers.Conv1D(filters=1,
                                                  kernel_size=1024,
                                                  trainable=False,
                                                  kernel_initializer=tf.keras.initializers.Ones,
                                                  padding="causal",
                                                  name="causal_convolution")(input)

        # storing all the skipped connexions from the residual blocs
        self.skipped = []
        residual = self.causal_conv
        for i in range(nb_residual):
            residual, skipped = self.__generate_block(nb_dilatation, residual)
            self.skipped.append(skipped)

        # processing the skip connexion
        skip_out = tf.keras.layers.Add(name="skip_connexion")(self.skipped)
        skip_out = tf.keras.layers.Activation('relu', name="RELU1_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Conv1D(filters=128,
                                          kernel_size=1,
                                          padding="same", name="CONV1_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Activation('relu', name="RELU2_SkipConnection")(skip_out)
        # as 256 is the codec of the audio
        # The model outputs a categorical distribution over the next value xt with a softmax layer
        skip_out = tf.keras.layers.Conv1D(filters=256,
                                          kernel_size=1,
                                          padding="same", name="CONV2_SkipConnection")(skip_out)
        outputWN = tf.keras.layers.Activation('softmax', name="SOFTMAX_SkipConnection")(skip_out)

        # defining the model
        self.model = tf.keras.Model(inputs=input,
                                    outputs=outputWN)

    def __generate_block(self, nb_dilatation, input):
        tanh_dilated_conv = input
        sigma_dilated_conv = input
        # how many dilated convolutions will be stacked
        for j in range(nb_dilatation):
            for i in range(10):
                # first dilated convolution, activated with tanh
                tanh_dilated_conv = tf.keras.layers.Conv1D(filters=1,
                                                           kernel_size=2,
                                                           padding="causal",
                                                           dilation_rate=2 ** i)(tanh_dilated_conv)

            for i in range(10):
                # second dilated convolution, activated with simoid
                sigma_dilated_conv = tf.keras.layers.Conv1D(filters=1,
                                                            kernel_size=2,
                                                            padding="causal",
                                                            dilation_rate=2 ** i)(sigma_dilated_conv)

        # activation
        tanh_out = tf.keras.layers.Activation("tanh")(tanh_dilated_conv)
        sigma_out = tf.keras.layers.Activation("sigmoid")(sigma_dilated_conv)

        # sigma and tanh multiply
        multiply = tf.keras.layers.Multiply()([tanh_out, sigma_out])
        conv11 = tf.keras.layers.Conv1D(filters=1,
                                        kernel_size=1,
                                        padding="same")(multiply)

        residual = tf.keras.layers.Add()([input, conv11])
        skip = tf.keras.layers.Conv1D(filters=128,
                                      kernel_size=1,
                                      padding="same")(multiply)

        return residual, skip
