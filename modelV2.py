import tensorflow as tf


class WavenetModel:

    def __init__(self, nb_layers, **kwargs):
        input = tf.keras.Input(name="WaveNet Input")
        self.causal_conv = tf.keras.layers.Conv1D(paddinf="causal",
                                                  name="causal_convolution")(input)

        self.skipped = []
        for i in range(nb_layers):
            residual, skipped = self.__generate_block(2)
            self.skipped.append(skipped)

        skip_out = tf.keras.layers.Add(name="skip connexion")(self.skipped)
        skip_out = tf.keras.layers.Activation('relu', name="RELU1_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Conv1D(padding="same", name="CONV1_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Activation('relu', name="RELU2_SkipConnection")(skip_out)
        skip_out = tf.keras.layers.Conv1D(padding="same", name="CONV2_SkipConnection")(skip_out)

        outputWN = tf.keras.layers.Activation('softmax', name="SOFTMAX_SkipConnection")(skip_out)

        self.model = tf.keras.Model(input=input,
                                    outputs=outputWN)

    def __generate_block(self, dilated_length):

        # first dilated convolution, activated with tanh
        tanh_dilated_conv = self.causal_conv
        for i in range(dilated_length - 1):
            tanh_dilated_conv = tf.keras.layers.Conv1D(padding="causal",
                                                       dilation_rate=2 ** i,
                                                       name="tanh_dilated_conv-{}".format(2 ** i))(tanh_dilated_conv)

        tanh_out = tf.keras.layers.Activation("tanh")(tanh_dilated_conv)

        # second dilated convolution, activated with simoid
        sigma_dilated_conv = self.causal_conv
        for i in range(dilated_length - 1):
            sigma_dilated_conv = tf.keras.layers.Conv1D(padding="causal",
                                                        dilation_rate=2 ** i,
                                                        name="sigma_dilated_conv-{}".format(2 ** i))(sigma_dilated_conv)

        sigma_out = tf.keras.layers.Activation("sigmoid")(sigma_dilated_conv)

        # sigma and tanh multiply
        multiply = tf.keras.layers.Multiply()([tanh_out, sigma_out])

        conv11 = tf.keras.layers.Conv1D(padding="causal",
                                        name="conv 1x1 residual")(multiply)

        residual = tf.keras.layers.Add()([self.causal_conv, conv11])
        skip = tf.keras.layers.Conv1D(padding="causal",
                                      name="conv 1x1 skipped")(multiply)

        return (residual, skip)
