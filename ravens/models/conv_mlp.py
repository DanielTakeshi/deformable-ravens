import tensorflow as tf; tf.compat.v1.enable_eager_execution()

import numpy as np
from tensorflow.keras import layers
import tensorflow_hub as hub

class ConvMLP(tf.keras.Model):
    def __init__(self, d_action, pretrained=True):
        super(ConvMLP, self).__init__()

        if pretrained:
            import tensorflow_hub as hub
            inception = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4",
                trainable=True)
            for i in inception.weights:
                if ('Conv2d_1a_7x7/weights') in i.name:
                    conv1weights = i
                    break

        self.d_action = d_action

        input_shape = (None, 320, 160, 3)

        if pretrained:
            self.conv1 = layers.Conv2D(filters=64, kernel_size=(7, 7),
                strides=(2,2), weights=[conv1weights.numpy(), tf.zeros(64)], input_shape=input_shape)

        else:
            self.conv1 = layers.Conv2D(filters=64, kernel_size=(7, 7),
                strides=(2,2), input_shape=input_shape)

        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters=32, kernel_size=(5, 5),
            strides=(1,1))
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(filters=16, kernel_size=(5, 5),
            strides=(1,1))
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1024, kernel_initializer="normal",
                            bias_initializer="normal",
                            activation='relu')
        self.drop1 = layers.Dropout(0.2)
        self.fc2 = layers.Dense(1024, kernel_initializer="normal",
                                bias_initializer="normal",
                                activation='relu')
        self.drop2 = layers.Dropout(0.2)
        self.fc3 = layers.Dense(d_action,
                                kernel_initializer="normal",
                                bias_initializer="normal")

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


    def compute_spatial_soft_argmax(self, x):
        """
        Parameter-less, extract coordinates for each channel.

        ~H = size related to original image H size
        ~W = size related to original image W size
        C channels

        Args:
          x, shape: (batch_size, ~H, ~W, C)
        Returns:
          shape: (batch_size, C, 2)
        """

        
        
        # unfortunately can't easily dynamically compute these sizes
        # inside a @tf.function, so just hard-coding them
        H, W, C = 149, 69, 16
        B = self.batch_size

        # see: https://github.com/tensorflow/tensorflow/issues/6271
        x = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [B * C, H * W])
        softmax = tf.nn.softmax(x)
        softmax = tf.transpose(tf.reshape(softmax, [B, C, H, W]), [0, 2, 3, 1])
        
        posx, posy = tf.meshgrid(tf.linspace(-1., 1., num=H), 
                                 tf.linspace(-1., 1., num=W), 
                                 indexing='ij')

        image_coords = tf.stack((posx, posy), axis=2) # (H, W, 2)
        # Convert softmax to shape [B, H, W, C, 1]
        softmax = tf.expand_dims(softmax, -1)
        # Convert image coords to shape [H, W, 1, 2]
        image_coords = tf.expand_dims(image_coords, 2)
        # Multiply (with broadcasting) and reduce over image dimensions to get the result
        # of shape [B, C, 2]
        spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, axis=[1, 2])
        return spatial_soft_argmax

    def call(self, x):
        """
        Args:
          x, shape: (batch_size, H, W, C)
        Return:
          shape: (batch_size, self.d_action)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        # shape (B, ~H, ~W, C=16)

        x = self.compute_spatial_soft_argmax(x) # shape (B, C, 2)

        x = self.flatten(x) # shape (B, C*2)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":

    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices('GPU')
    MEM_LIMIT = 1024 * 4
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=MEM_LIMIT)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    conv_mlp = ConvMLP(d_action=3)

    img = np.random.randn(7, 320, 160, 3)
    out = conv_mlp(img)
    print(out.shape)
    
