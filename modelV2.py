import tensorflow as tf


class WavenetModel:

    def __init__(self,nb_residual, nb_layers, nb_dilatation, audio_length, **kwargs):
        input = tf.keras.Input(shape=(audio_length, 1), name="WaveNet_Input")
        self.causal_conv = tf.keras.layers.Conv1D(filters=nb_residual,
                                                  kernel_size=1,
                                                  padding="causal",
                                                  name="causal_convolution")(input)

        self.skipped = []
        for i in range(nb_residual):
            residual, skipped = self.__generate_block(nb_dilatation)
            self.skipped.append(skipped)

        skip_out = tf.keras.layers.Add(name="skip_connexion")(self.skipped)
        skip_out = tf.keras.layers.Activation('relu', name="RELU1_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Conv1D(filters=1,
                                          kernel_size=2,
                                          padding="same", name="CONV1_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Activation('relu', name="RELU2_SkipConnection")(skip_out)
        # as 256 is the codec of the audio
        # The model outputs a categorical distribution over the next value xt with a softmax layer
        skip_out = tf.keras.layers.Conv1D(filters=256,
                                          kernel_size=1,
                                          padding="same", name="CONV2_SkipConnection")(skip_out)
        outputWN = tf.keras.layers.Activation('softmax', name="SOFTMAX_SkipConnection")(skip_out)

        self.model = tf.keras.Model(inputs=input,
                                    outputs=outputWN)

    def __generate_block(self, nb_dilatation):
        tanh_dilated_conv = self.causal_conv
        sigma_dilated_conv = self.causal_conv
        for j in range(nb_dilatation):
            # first dilated convolution, activated with tanh
            for i in range(10):
                tanh_dilated_conv = tf.keras.layers.Conv1D(filters=1,
                                                           kernel_size=2,
                                                           padding="causal",
                                                           dilation_rate=2 ** i)(tanh_dilated_conv)

            # second dilated convolution, activated with simoid
            for i in range(10):
                sigma_dilated_conv = tf.keras.layers.Conv1D(filters=1,
                                                            kernel_size=2,
                                                            padding="causal",
                                                            dilation_rate=2 ** i)(sigma_dilated_conv)

        tanh_out = tf.keras.layers.Activation("tanh")(tanh_dilated_conv)
        sigma_out = tf.keras.layers.Activation("sigmoid")(sigma_dilated_conv)

        # sigma and tanh multiply
        multiply = tf.keras.layers.Multiply()([tanh_out, sigma_out])
        conv11 = tf.keras.layers.Conv1D(filters=1,
                                        kernel_size=1,
                                        padding="same")(multiply)

        residual = tf.keras.layers.Add()([self.causal_conv, conv11])
        skip = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=1,
                                      padding="same")(multiply)

        return (residual, skip)
