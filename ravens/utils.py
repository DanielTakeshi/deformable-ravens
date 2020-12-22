#!/usr/bin/env python

import sys
import time
import struct
import threading

import cv2
import numpy as np
import pybullet as p
import matplotlib
import matplotlib.pyplot as plt

import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import transformations

#-----------------------------------------------------------------------------
# HEIGHTMAP UTILS
#-----------------------------------------------------------------------------

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[..., 0] >= bounds[0, 0]) & (points[..., 0] < bounds[0, 1])
    iy = (points[..., 1] >= bounds[1, 0]) & (points[..., 1] < bounds[1, 1])
    iz = (points[..., 2] >= bounds[2, 0]) & (points[..., 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[..., i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds.

    The color and depth are np.arrays or lists, where the leading dimension
    or list wraps around differnet viewpoints. So, if the np.array shape is
    (3,480,640,3), then the leading '3' denotes the number of camera views.

    TODO: documentation.
    """
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config['position']).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps


def pixel_to_position(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return (x, y, z)


def position_to_pixel(position, bounds, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return (u, v)


def unproject_vectorized(uv_coordinates: np.ndarray, depth_values: np.ndarray,
                         intrinsic: np.ndarray,
                         distortion: np.ndarray) -> np.ndarray:
  """Vectorized version of unproject(), for N points.

  Args:
    uv_coordinates: pixel coordinates to unproject of shape (n, 2).
    depth_values: depth values corresponding index-wise to the uv_coordinates of
      shape (n).
    intrinsic: array of shape (3, 3). This is typically the return value
      of intrinsics_to_matrix.
    distortion: camera distortion parameters of shape (5,).

  Returns:
    xyz coordinates in camera frame of shape (n, 3).
  """
  cam_mtx = intrinsic  # shape [3, 3]
  cam_dist = np.array(distortion)  # shape [5]

  # shape of points_undistorted is [N, 2] after the squeeze().
  points_undistorted = cv2.undistortPoints(
      uv_coordinates.reshape((-1, 1, 2)), cam_mtx, cam_dist).squeeze()

  x = points_undistorted[:, 0] * depth_values
  y = points_undistorted[:, 1] * depth_values

  xyz = np.vstack((x, y, depth_values)).T
  return xyz


def unproject_depth_vectorized(im_depth: np.ndarray, depth_dist: np.ndarray,
                               camera_mtx: np.ndarray,
                               camera_dist: np.ndarray) -> np.ndarray:
  """Unproject depth image into 3D point cloud, using calibration.

  Args:
    im_depth: raw depth image, pre-calibration of shape (height, width).
    depth_dist: depth distortion parameters of shape (8,)
    camera_mtx: intrinsics matrix of shape (3, 3). This is typically the return
      value of intrinsics_to_matrix.
    camera_dist: camera distortion parameters shape (5,).

  Returns:
    numpy array of shape [3, H*W]. each column is xyz coordinates
  """
  h, w = im_depth.shape

  # shape of each u_map, v_map is [H, W].
  u_map, v_map = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))

  adjusted_depth = depth_dist[0] + im_depth * depth_dist[1]

  # shape after stack is [N, 2], where N = H * W.
  uv_coordinates = np.stack((u_map.reshape(-1), v_map.reshape(-1)), axis=-1)

  return unproject_vectorized(uv_coordinates, adjusted_depth.reshape(-1),
                              camera_mtx, camera_dist)

#-----------------------------------------------------------------------------
# MATH UTILS
#-----------------------------------------------------------------------------

def sample_distribution(prob, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = np.ndarray.flatten(prob) / np.sum(prob)
    rand_ind = np.random.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())

#-------------------------------------------------------------------------
# Transformation Helper Functions (Daniel: copied over from their code,
# however the first three are in task.py and that's the code we use.)
#-------------------------------------------------------------------------

def invert(pose):
    return p.invertTransform(pose[0], pose[1])


def multiply(pose0, pose1):
    return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])


def apply(pose, position):
    position = np.float32(position)
    position_shape = position.shape
    position = np.float32(position).reshape(3, -1)
    rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
    translation = np.float32(pose[0]).reshape(3, 1)
    position = rotation @ position + translation
    return tuple(position.reshape(position_shape))


def get_pybullet_quaternion_from_rot(rotation):
    """Abstraction for converting from a 3-parameter rotation to quaterion.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      rotation: a 3-parameter rotation, in xyz order tuple of 3 floats
    Returns:
      quaternion, in xyzw order, tuple of 4 floats
    """
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = transformations.quaternion_from_euler(*euler_zxy, axes='szxy')
    q = quaternion_wxyz
    quaternion_xyzw = (q[1], q[2], q[3], q[0])
    return quaternion_xyzw


def get_rot_from_pybullet_quaternion(quaternion_xyzw):
    """Abstraction for converting from quaternion to a 3-parameter toation.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      quaternion, in xyzw order, tuple of 4 floats
    Returns:
      rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
    """
    q = quaternion_xyzw
    quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
    euler_zxy = transformations.euler_from_quaternion(quaternion_wxyz, axes='szxy')
    euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
    return euler_xyz


def apply_transform(transform_to_from, points_from):
  r"""Transforms points (3D) into new frame.

  Using transform_to_from notation.

  Args:
    transform_to_from: numpy.ndarray of shape [B,4,4], SE3
    points_from: numpy.ndarray of shape [B,3,N]

  Returns:
    points_to: numpy.ndarray of shape [B,3,N]
  """
  num_points = points_from.shape[-1]

  # non-batched
  if len(transform_to_from.shape) == 2:
    ones = np.ones((1, num_points))

    # makes these each into homogenous vectors
    points_from = np.vstack((points_from, ones))  # [4,N]
    points_to = transform_to_from @ points_from  # [4,N]
    return points_to[0:3, :]  # [3,N]

  # batched
  else:
    assert len(transform_to_from.shape) == 3
    batch_size = transform_to_from.shape[0]
    zeros = np.ones((batch_size, 1, num_points))
    points_from = np.concatenate((points_from, zeros), axis=1)
    assert points_from.shape[1] == 4
    points_to = transform_to_from @ points_from
    return points_to[:, 0:3, :]

#-----------------------------------------------------------------------------
# IMAGE UTILS
#-----------------------------------------------------------------------------

def preprocess_color(image, mean=0.5, std=0.225):
    image = (image.copy() / 255 - mean) / std
    return image


def preprocess_depth(image, mean=0.005, std=0.008):
    image = (image.copy() - mean) / std
    image = np.tile(image.reshape(
        image.shape[0], image.shape[1], 1), (1, 1, 3))
    return image


def get_image_transform(theta, trans, pivot=[0, 0]):
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_T_image = np.array([[1., 0., -pivot[0]],
                              [0., 1., -pivot[1]],
                              [0., 0.,        1.]])
    image_T_pivot = np.array([[1., 0., pivot[0]],
                              [0., 1., pivot[1]],
                              [0., 0.,       1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]],
                          [0.,            0.,            1.]])
    return np.dot(image_T_pivot, np.dot(transform, pivot_T_image))


def check_transform(image, pixel, transform):
    """Valid transform only if pixel locations are still in FoV after transform."""
    new_pixel = np.flip(np.int32(np.round(np.dot(transform, np.float32(
        [pixel[1], pixel[0], 1.]).reshape(3, 1))))[:2].squeeze())
    valid = np.all(new_pixel >= 0) and new_pixel[0] < image.shape[
        0] and new_pixel[1] < image.shape[1]
    return valid, new_pixel


def get_se3_from_image_transform(theta, trans, pivot, heightmap, bounds, pixel_size):
    position_center = pixel_to_position(np.flip(np.int32(np.round(pivot))),
        heightmap, bounds, pixel_size, skip_height=False)
    new_position_center = pixel_to_position(np.flip(np.int32(np.round(pivot + trans))),
        heightmap, bounds, pixel_size, skip_height=True)
    # Don't look up the z height, it might get augmented out of frame
    new_position_center = (new_position_center[0], new_position_center[1], position_center[2])

    delta_position = np.array(new_position_center) - np.array(position_center)

    t_world_center = np.eye(4)
    t_world_center[0:3, 3] = np.array(position_center)

    t_centernew_center = np.eye(4)
    euler_zxy = (-theta, 0, 0)
    t_centernew_center[0:3, 0:3] = transformations.euler_matrix(*euler_zxy, axes='szxy')[0:3, 0:3]

    t_centernew_center_Tonly = np.eye(4)
    t_centernew_center_Tonly[0:3, 3] = - delta_position
    t_centernew_center = t_centernew_center @ t_centernew_center_Tonly

    t_world_centernew = t_world_center @ np.linalg.inv(t_centernew_center)
    return t_world_center, t_world_centernew


def get_random_image_transform_params(image_size):
    theta_sigma = 2 * np.pi / 6
    theta = np.random.normal(0, theta_sigma)

    trans_sigma = np.min(image_size) / 6
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot


# -------------------------------------------------------------------------------------- #
# Daniel: this is slightly different in their updated code, where they return rounded
# pixels. After carefully checking their updated code, their `rounded_pixels` corresponds
# to the OLD way of creating `pixels`. We originally returned `input_image, new_pixels`,
# and the new pixels were already rounded. Later, they must have wanted to return the
# non-rounded versions, so they re-interpret `pixels` to be the non-rounded version, and
# added `rounded_pixels` to make the distinction more explicit.
# -------------------------------------------------------------------------------------- #
# Their code also calls `theta, trans, pivot = get_random_image_transform_params`. We use
# the EXACT code; I assume they wanted to call it from `agents/gt_state.py`, because that
# agent should use the same transformation, except it doesn't use an input image. Thus, it
# just needs the parameters so that it can change all the ground-truth poses.
# -------------------------------------------------------------------------------------- #
# Code gets called from agents/{conv_mlp,form2fit,transporter}.py. We only have transporter
# here, and in all cases, set_theta_zero=False by default so that's fine. Finally, consider
# return values. Their transporter.py uses `transform_params` but ONLY for the 6 DoF agent,
# thus we don't need to return it. Their code also doesn't even return the non-rounded
# pixels, so again it should be safe to ignore. :)
# -------------------------------------------------------------------------------------- #

def perturb(input_image, pixels, set_theta_zero=False):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size)
        if set_theta_zero:
            theta = 0.
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)
            pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            pixel = np.flip(pixel)
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]
            is_valid = is_valid and np.all(pixel >= 0) and in_fov
            new_pixels.append(pixel)
        if is_valid:
            break

    # Apply rigid transform to image and pixel labels.
    input_image = cv2.warpAffine(input_image, transform[:2, :],
                                 (image_size[1], image_size[0]),
                                 flags=cv2.INTER_NEAREST)
    return input_image, new_pixels

#-----------------------------------------------------------------------------
# PLOT UTILS
#-----------------------------------------------------------------------------

# Plot colors (Tableau palette).
COLORS = {'blue':   [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
          'red':    [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
          'green':  [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
          'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
          'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
          'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
          'pink':   [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
          'cyan':   [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
          'brown':  [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
          'gray':   [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0]}


def plot(fname, title, ylabel, xlabel, data, xlim=[-np.inf, 0], xticks=None, ylim=[np.inf, -np.inf], show_std=True):
    # Data is a dictionary that maps experiment names to tuples with 3
    # elements: x (size N array) and y (size N array) and y_std (size N array)

    # Get data limits.
    for name, (x, y, _) in data.items():
        y = np.array(y)
        xlim[0] = max(xlim[0], np.min(x))
        xlim[1] = max(xlim[1], np.max(x))
        ylim[0] = min(ylim[0], np.min(y))
        ylim[1] = max(ylim[1], np.max(y))

    # Draw background.
    plt.title(title, fontsize=14)
    plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xlim)
    plt.xlabel(xlabel, fontsize=14)
    plt.grid(True, linestyle='-', color=[0.8, 0.8, 0.8])
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('#000000')
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Draw data.
    color_iter = 0
    for name, (x, y, std) in data.items():
        x, y, std = np.float32(x), np.float32(y), np.float32(std)
        upper = np.clip(y + std, ylim[0], ylim[1])
        lower = np.clip(y - std, ylim[0], ylim[1])
        color = COLORS[list(COLORS.keys())[color_iter]]
        if show_std:
            plt.fill_between(x, upper, lower, color=color,
                             linewidth=0, alpha=0.3)
        plt.plot(x, y, color=color,
                 linewidth=2, marker='o', alpha=1.)
        color_iter += 1

    if xticks:
        plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=14)
    else:
        plt.xticks(fontsize=14)
    plt.legend([name for name, _ in data.items()],
               loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()

#-----------------------------------------------------------------------------
# MESHCAT UTILS
#-----------------------------------------------------------------------------

def create_visualizer(clear=True):
    print("Waiting for meshcat server... have you started a server?")
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    return vis


def make_frame(vis, name, h, radius, o=1.0):
  """Add a red-green-blue triad to the Meschat visualizer.

  Args:
    vis (MeshCat Visualizer): the visualizer
    name (string): name for this frame (should be unique)
    h (float): height of frame visualization
    radius (float): radius of frame visualization
    o (float): opacity

  """
  vis[name]["x"].set_object(g.Cylinder(height=h, radius=radius),
                            g.MeshLambertMaterial(color=0xff0000,
                                                  reflectivity=0.8,
                                                  opacity=o))
  rotate_x = mtf.rotation_matrix(np.pi/2.0, [0, 0, 1])
  rotate_x[0, 3] = h/2
  vis[name]["x"].set_transform(rotate_x)

  vis[name]["y"].set_object(g.Cylinder(height=h, radius=radius),
                            g.MeshLambertMaterial(color=0x00ff00,
                                                  reflectivity=0.8,
                                                  opacity=o))
  rotate_y = mtf.rotation_matrix(np.pi/2.0, [0, 1, 0])
  rotate_y[1, 3] = h/2
  vis[name]["y"].set_transform(rotate_y)

  vis[name]["z"].set_object(g.Cylinder(height=h, radius=radius),
                            g.MeshLambertMaterial(color=0x0000ff,
                                                  reflectivity=0.8,
                                                  opacity=o))
  rotate_z = mtf.rotation_matrix(np.pi/2.0, [1, 0, 0])
  rotate_z[2, 3] = h/2
  vis[name]["z"].set_transform(rotate_z)


def meshcat_visualize(vis, obs, act, info):

    for key in sorted(info.keys()):

        pose = info[key]
        pick_transform = np.eye(4)
        pick_transform[0:3,3] = pose[0]
        quaternion_wxyz = np.asarray([pose[1][3], pose[1][0], pose[1][1], pose[1][2]])
        pick_transform[0:3,0:3] = mtf.quaternion_matrix(quaternion_wxyz)[0:3,0:3]
        label = "obj_"+str(key)
        make_frame(vis, label, h=0.05, radius=0.0012, o=1.0)
        vis[label].set_transform(pick_transform)

    for cam_index in range(len(act['camera_config'])):

        verts = unproject_depth_vectorized(obs['depth'][cam_index],
            [0,1], np.array(act['camera_config'][cam_index]['intrinsics']).reshape(3,3),
            np.zeros(5))

        # switch from [N,3] to [3,N]
        verts = verts.T

        cam_transform = np.eye(4)
        cam_transform[0:3,3] = act['camera_config'][cam_index]['position']
        quaternion_xyzw = act['camera_config'][cam_index]['rotation']
        quaternion_wxyz = np.asarray([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
        cam_transform[0:3,0:3] = mtf.quaternion_matrix(quaternion_wxyz)[0:3,0:3]
        verts = apply_transform(cam_transform, verts)

        colors = obs['color'][cam_index].reshape(-1, 3).T / 255.0

        vis["pointclouds/"+str(cam_index)].set_object(g.PointCloud(position=verts,
            color=colors))

#-----------------------------------------------------------------------------
# Daniel's misc utils
#-----------------------------------------------------------------------------

def round_pos(pos, dig=3):
    """Make it a little easier to read from debug prints."""
    pos = (round(pos[0], dig), round(pos[1], dig), round(pos[2], dig))
    return pos


def round_orn(orn, dig=3):
    """Make it a little easier to read from debug prints."""
    orn = (round(orn[0], dig), round(orn[1], dig), round(orn[2], dig), round(orn[3], dig))
    return orn


def round_pose(pose, dig=3):
    """Make it a little easier to read from debug prints."""
    return (round_pos(pose[0]), round_orn(pose[1]))


def print_dict(info):
    for key in info:
        print('  {}  -->  {}'.format(key, info[key]))
    print()


class TrackIDs:
    """Might help me track all of PyBullet's IDs a little better."""

    def __init__(self):
        self._id_to_name = {}

    def add(self, ID, name):
        assert ID not in self._id_to_name, self._id_to_name
        assert isinstance(ID, int), ID
        self._id_to_name[ID] = name

    def __str__(self):
        string = "\nID Tracker. ID to Name:\n"
        ints = sorted(self._id_to_name.keys())
        for id in ints:
            string += "{} --> {}\n".format(str(id).zfill(2), self._id_to_name[id])
        return string


def fit_circle(points_l, scale, debug=False):
    """Get information about a circle from a list of points `points_l`.

    This may involve fitting a circle or ellipse to a set of points?

    pip install circle-fit
    https://github.com/AlliedToasters/circle-fit

    Assuing for now that points_l contains a list of (x,y,z) points, so we
    take only (x,y) and scale according to `scale`. Both methods return a
    tuple of four values:

    xc: x-coordinate of solution center (float)
    yc: y-coordinate of solution center (float)
    R: Radius of solution (float)
    variance or residual (float)

    These methods should be identical if we're querying this with actual
    circles. Returning the second one for now.
    """
    from circle_fit import hyper_fit, least_squares_circle
    data = [ (item[0]*scale, item[1]*scale) for item in points_l ]
    data = np.array(data)
    circle_1 = hyper_fit(data)
    circle_2 = least_squares_circle(data)
    xc_1, yc_1, r_1, _ = circle_1
    xc_2, yc_2, r_2, _ = circle_2
    if debug:
        print(f'(hyperfit) rad {r_1:0.4f}, center ({xc_1:0.4f},{yc_1:0.4f})')
        print(f'(least-sq) rad {r_2:0.4f}, center ({xc_2:0.4f},{yc_2:0.4f})')
    return circle_2
