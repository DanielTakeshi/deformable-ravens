import tensorflow as tf; tf.compat.v1.enable_eager_execution()
import numpy as np
from tensorflow.keras import layers


class MlpModel(tf.keras.Model):

    def __init__(self, batch_size, d_obs, d_action, activation='relu', mdn=False,
                 dropout=0.2, use_sinusoid=True):
        """MLP model, with Mixture Density Networks (MDNs) supported.

        Daniel: looks like (obs_dim -> 128 -> 128 -> act_dim), roughly, when running
        the non-MDN case and without the sinusoid stuff. The MDN case (which we should
        be using) will do this:

            (obs_dim*3) -> (128) -> (128) -> (d_act*m, m, m)

        Note I: we have m=26 by default, so it's a 26-dimensional mixture model.

        Note II: with MDN, there are three outputs given the input. The first provides
        the mean of all the Gaussians, so it's d_act times m: each of the 26 modes has
        its own mean. But, the log(variance) is assumed to be similar among all the
        means. That's a fair simplifying assumption, hence we just need an m-dimensional
        vector. Finally, the last m-dimensional vector is for the mixing coefficients,
        to represent probabilities among the modes. I.e., given this mixing vector, it's
        a discrete m-way probability distribution, and it gets sampled to determine the
        distribution to use, THEN we draw from the corresponding Gaussian.

        Note III: it uses sinusoids: z <-- [z, sin(z), cos(z)], as explained in the
        CoRL submission appendix, since NeRF used it. Strangely, the normalization is
        commented out, so it's not used? Maybe it's irrelevant with sinusoids?

        Note IV: in reality, the observation gets passed as input again to the second
        and the third FC layers. Presumably that must have helped.
        """
        super(MlpModel, self).__init__()
        self.normalize_input = True

        self.use_sinusoid = use_sinusoid
        if self.use_sinusoid:
            k = 3
        else:
            k = 1

        self.fc1 = layers.Dense(128, input_shape=(batch_size, d_obs*k),
                                kernel_initializer="normal",
                                bias_initializer="normal",
                                activation=activation)
        self.drop1 = layers.Dropout(dropout)
        self.fc2 = layers.Dense(128, kernel_initializer="normal",
                                bias_initializer="normal",
                                activation=activation)
        self.drop2 = layers.Dropout(dropout)
        self.fc3 = layers.Dense(d_action,
                                kernel_initializer="normal",
                                bias_initializer="normal")

        self.mdn = mdn
        if self.mdn:
            k = 26
            self.mu = tf.keras.layers.Dense((d_action * k),
                                             kernel_initializer="normal",
                                             bias_initializer="normal")
            # Variance should be non-negative, so exp()
            self.logvar = tf.keras.layers.Dense(k,
                                                kernel_initializer="normal",
                                                bias_initializer="normal")

            # mixing coefficient should sum to 1.0, so apply softmax
            self.pi = tf.keras.layers.Dense(k,
                                            kernel_initializer="normal",
                                            bias_initializer="normal")
            self.softmax = tf.keras.layers.Softmax()
            self.temperature = 2.5

    def reset_states(self):
        pass

    def set_normalization_parameters(self, obs_train_parameters):
        """
        obs_train_parameters: dict with key, values:
          'mean', numpy.ndarray of shape (obs_dimension)
          'std', numpy.ndarray of shape (obs_dimension)
        """
        self.obs_train_mean = obs_train_parameters['mean']
        self.obs_train_std = obs_train_parameters['std']

    def call(self, x):
        """
        Args:
            x, shape: (batch_size, obs_dimension)

        Return:
            (if not MDN)
            shape: (batch_size, action_dimension)

            (if MDN)
            shape of pi: (batch_size, num_gaussians)
            shape of mu: (batch_size, num_gaussians*action_dimension)
            shape of var: (batch_size, num_gaussians)
        """
        obs = x*1.0
        #if self.normalize_input:
        #    x = (x - self.obs_train_mean) / (self.obs_train_std + 1e-7)

        def cs(x):
            if self.use_sinusoid:
                sin = tf.math.sin(x)
                cos = tf.math.cos(x)
                return tf.concat((x, cos, sin), axis=1)
            else:
                return x

        x = self.drop1(self.fc1(cs(obs)))
        x = self.drop2(self.fc2(tf.concat((x, cs(obs)), axis=1)))
        x = tf.concat((x, cs(obs)), axis=1)

        if not self.mdn:
            x = self.fc3(x)
            return x
        else:
            pi = self.pi(x)
            pi = pi / self.temperature
            pi = self.softmax(pi)
            mu = self.mu(x)
            var = tf.math.exp(self.logvar(x))
            return (pi, mu, var)
