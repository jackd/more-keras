from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def polar_to_cartesian(dist, theta, phi):
    z = np.cos(phi)
    s = np.sin(phi)
    x = s * np.cos(theta)
    y = s * np.sin(theta)
    return np.stack((x, y, z), axis=-1) * dist


def normalized(vec):
    return vec / np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)


def _look_at_nh_helper(eye, center=None, world_up=None, dtype=None, axis=-1):
    """Non-homogeneous eye-to-world coordinate transform."""
    # vector_degeneracy_cutoff = 1e-6
    eye = np.array(eye, dtype=dtype)
    if center is None:
        center = np.zeros_like(eye)
    else:
        center = np.array(center, dtype=dtype)
    if world_up is None:
        world_up = np.zeros_like(eye)
        world_up[..., 2] = 1  # pylint: disable=unsupported-assignment-operation
    else:
        world_up = np.array(world_up, dtype=dtype)

    # https://web.cs.wpi.edu/~emmanuel/courses/cs543/f13/slides/lecture04_p3.pdf
    n = normalized(eye - center)
    u = normalized(np.cross(world_up, n))
    v = np.cross(n, u)
    v = normalized(v)
    rotation = np.stack([u, v, n], axis=axis)

    translation = eye
    return rotation, translation


def look_at_nh(eye, center=None, world_up=None, dtype=None):
    R, t = _look_at_nh_helper(eye, center, world_up, dtype, axis=-2)
    t = np.matmul(R, -t)
    return R, t


def inverse_look_at_nh(eye, center=None, world_up=None, dtype=None):
    return _look_at_nh_helper(eye, center, world_up, dtype, axis=-1)


def as_homogeneous_transform(transform, t=None):
    out = np.zeros((4, 4), dtype=transform.dtype)
    out[:3, :3] = transform
    if t is not None:
        out[:3, 3] = t
    out[3, 3] = 1
    return out
