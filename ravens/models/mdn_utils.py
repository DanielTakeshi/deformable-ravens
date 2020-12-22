import tensorflow as tf; tf.compat.v1.enable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt


EPS = 1e-12
  
def pick_max_mean(pi, mu, var):
  """Prediction as the mean of the most-weighted gaussian.

  Args are all TF:
    pi: (batch_size, num_gaussians)
    mu: (batch_size, num_gaussians * d_out)
    var: (batch_size, num_gaussians)
  Returns:
    (batch_size, d_out) NUMPY
  """
  mu = tf.reshape(mu, (tf.shape(mu)[0], tf.shape(pi)[1], -1))
  d_out = tf.shape(mu)[-1]
  batch_size, k = pi.shape
  prediction = np.zeros((batch_size, d_out))
  argmax_pi = tf.argmax(pi, axis=1) # shape (batch_size)
  for i in range(batch_size):
    ith_argmax_pi = argmax_pi[i].numpy()
    prediction[i] = mu[i, ith_argmax_pi]
  return prediction

def sample_from_pdf(pi, mu, var, num_samples=1):
  """Prediction as a sample from the gaussian mixture.

  Args are all TF:
    pi: (batch_size, num_gaussians)
    mu: (batch_size, num_gaussians * d_out)
    var: (batch_size, num_gaussians)
  Returns:
    (batch_size, num_samples, d_out) NUMPY
  """
  pi, mu, var = pi.numpy(), mu.numpy(), var.numpy()
  # apply temperature?
  #pi = pi**4 # apply temp
  var = var**4

  pi = pi * (1/pi.sum(1)[..., None])
  batch_size, k = pi.shape

  mu = tf.reshape(mu, (tf.shape(mu)[0], tf.shape(pi)[1], -1))
  d_out = tf.shape(mu)[-1]

  samples = np.zeros((batch_size, num_samples, d_out))
  for i in range(batch_size):
    for j in range(num_samples):
      idx = np.random.choice(range(k), p=pi[i])
      draw = np.random.normal(mu[i, idx], np.sqrt(var[i, idx]))
      samples[i,j] = draw
  return samples   


def multivar_gaussian_pdf(y, mu, var):
  r"""
  Assumes covariance matrix is identity times variance, i.e:
    \Sigma = I \sigma^2
    for \Sigma covariance matrix, \sigma std. deviation.

  Args:
    y: shape (batch_size, d)
    mu: shape (batch_size, k, d)
    var: shape (batch_size, k)
  Returns:   
  """
  # assert len(y.shape) == 2
  # assert len(mu.shape) == 3
  # assert len(var.shape) == 2
  # assert tf.shape(y)[-1] == tf.shape(mu)[-1]
  # assert tf.shape(mu)[1] == tf.shape(var)[-1]
  # assert tf.shape(y)[0] == tf.shape(mu)[0]
  # assert tf.shape(y)[0] == tf.shape(var)[0]
  y = tf.expand_dims(y, 1)
  d = mu.shape[-1]
  dot_prod = tf.reduce_sum((y - mu)**2, (2))  # shape (batch_size, k)
  exp_factor = tf.math.divide_no_nan(-1., (2. * (var))) *  dot_prod
  numerator = tf.math.exp(exp_factor)  # shape (batch_size, k)
  denominator =  tf.math.sqrt( (2 * np.pi * (var)) ** d)
  return tf.math.multiply_no_nan(numerator, 1/denominator)


def mdn_loss(y, mdn_predictions):
  """
  Args:
    y: true "y", shape (batch_size, d_out)
    mdn_predictions: tuple of:
      pi: (batch_size, num_gaussians)
      mu: (batch_size, num_gaussians * d_out)
      var: (batch_size, num_gaussians)
  Returns:
    loss, scalar
  """
  pi, mu, var = mdn_predictions
  mu = tf.reshape(mu, (tf.shape(mu)[0], tf.shape(pi)[-1], -1))
  # mu now (batch_size, num_gaussians, d_out) shape
  pdf = multivar_gaussian_pdf(y, mu, var)
  # multiply with each pi and sum it
  p = tf.multiply(tf.clip_by_value(pdf,1e-8,1e8), tf.clip_by_value(pi,1e-8,1e8))
  p = tf.reduce_sum(p, axis=1, keepdims=True)
  p = -tf.math.log(tf.clip_by_value(p,1e-8,1e8))
  #plot_mdn_predictions(y, mdn_predictions)
  return tf.reduce_mean(p)


#fig, ax = plt.subplots(1,1)

def plot_mdn_predictions(y, mdn_predictions):
  """
  Args:
    y: true "y", shape (batch_size, d_out)
    mdn_predictions: tuple of:
        pi: (batch_size, num_gaussians)
        mu: (batch_size, num_gaussians * d_out)
        var: (batch_size, num_gaussians)
  """
  pi, mu, var = mdn_predictions

  n = 5
  y = y[:n,:]
  pi = pi[:n,:]
  mu = mu[:n,:]
  var = var[:n,:]

  ax.cla()
  ax.scatter(y[:,0], y[:,1])
  mu = tf.reshape(mu, (-1, y.shape[-1]))
  pi = tf.reshape(pi, (-1,))
  pi = tf.clip_by_value(pi, 0.01, 1.0)

  rgba_colors = np.zeros((len(pi),4))
  # for red the first column needs to be one
  rgba_colors[:,0] = 1.0
  # the fourth column needs to be your alphas
  rgba_colors[:, 3] = pi

  ax.scatter(mu[:,0], mu[:,1], color=rgba_colors)

  plt.draw()
  plt.pause(0.001)
