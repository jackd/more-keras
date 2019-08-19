from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def normalized(vec):
    return vec / tf.norm(vec, ord='euclidean', axis=-1, keepdims=True)


def _look_at_nh_helper(eye, center=None, world_up=None, dtype=None, axis=-1):
    """Non-homogeneous eye-to-world coordinate transform."""
    # vector_degeneracy_cutoff = 1e-6
    eye = tf.convert_to_tensor(eye, dtype=dtype)
    batch_size = tf.shape(eye)[0]
    if center is None:
        center = tf.zeros((batch_size, 3), dtype=tf.float32)
    else:
        center = tf.convert_to_tensor(center, dtype=dtype)
    if world_up is None:
        world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)
        world_up = tf.tile(world_up, (batch_size, 1))
    else:
        world_up = tf.convert_to_tensor(world_up, dtype=dtype)

    # forward = normalized(center - eye)
    # to_side = normalized(tf.cross(forward, world_up))
    # cam_up = normalized(tf.cross(to_side, forward))
    # rotation = tf.stack([to_side, cam_up, -forward], axis=axis)
    # [batch_size, 3, 3] matrix

    # https://web.cs.wpi.edu/~emmanuel/courses/cs543/f13/slides/lecture04_p3.pdf
    n = normalized(eye - center)
    u = normalized(tf.cross(world_up, n))
    v = tf.cross(n, u)
    v = normalized(v)
    rotation = tf.stack([u, v, n], axis=axis)

    translation = eye
    return rotation, translation


def look_at_nh(eye, center=None, world_up=None, dtype=None):
    with tf.name_scope('look_at_nh_transform'):
        R, t = _look_at_nh_helper(eye, center, world_up, dtype, axis=-2)
        t = tf.einsum('ijk,ik->ij', R, -t)  # pylint: disable=invalid-unary-operand-type
    return R, t


def inverse_look_at_nh(eye, center=None, world_up=None, dtype=None):
    with tf.name_scope('inverse_look_at_nh_transform'):
        R, t = _look_at_nh_helper(eye, center, world_up, dtype, axis=-1)
    return R, t
