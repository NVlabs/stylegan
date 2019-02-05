# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Perceptual Path Length (PPL)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)

#----------------------------------------------------------------------------

class PPL(metric_base.MetricBase):
    def __init__(self, num_samples, epsilon, space, sampling, minibatch_per_gpu, **kwargs):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph.
        distance_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                noise_vars = [var for name, var in Gs_clone.components.synthesis.vars.items() if name.startswith('noise')]

                # Generate random latents and interpolation t-values.
                lat_t01 = tf.random_normal([self.minibatch_per_gpu * 2] + Gs_clone.input_shape[1:])
                lerp_t = tf.random_uniform([self.minibatch_per_gpu], 0.0, 1.0 if self.sampling == 'full' else 0.0)

                # Interpolate in W or Z.
                if self.space == 'w':
                    dlat_t01 = Gs_clone.components.mapping.get_output_for(lat_t01, None, is_validation=True)
                    dlat_t0, dlat_t1 = dlat_t01[0::2], dlat_t01[1::2]
                    dlat_e0 = tflib.lerp(dlat_t0, dlat_t1, lerp_t[:, np.newaxis, np.newaxis])
                    dlat_e1 = tflib.lerp(dlat_t0, dlat_t1, lerp_t[:, np.newaxis, np.newaxis] + self.epsilon)
                    dlat_e01 = tf.reshape(tf.stack([dlat_e0, dlat_e1], axis=1), dlat_t01.shape)
                else: # space == 'z'
                    lat_t0, lat_t1 = lat_t01[0::2], lat_t01[1::2]
                    lat_e0 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis])
                    lat_e1 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis] + self.epsilon)
                    lat_e01 = tf.reshape(tf.stack([lat_e0, lat_e1], axis=1), lat_t01.shape)
                    dlat_e01 = Gs_clone.components.mapping.get_output_for(lat_e01, None, is_validation=True)

                # Synthesize images.
                with tf.control_dependencies([var.initializer for var in noise_vars]): # use same noise inputs for the entire minibatch
                    images = Gs_clone.components.synthesis.get_output_for(dlat_e01, is_validation=True, randomize_noise=False)

                # Crop only the face region.
                c = int(images.shape[2] // 8)
                images = images[:, :, c*3 : c*7, c*2 : c*6]

                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                if images.shape[2] > 256:
                    factor = images.shape[2] // 256
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3,5])

                # Scale dynamic range from [-1,1] to [0,255] for VGG.
                images = (images + 1) * (255 / 2)

                # Evaluate perceptual distance.
                img_e0, img_e1 = images[0::2], images[1::2]
                distance_measure = misc.load_pkl('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2') # vgg16_zhang_perceptual.pkl
                distance_expr.append(distance_measure.get_output_for(img_e0, img_e1) * (1 / self.epsilon**2))

        # Sampling loop.
        all_distances = []
        for _ in range(0, self.num_samples, minibatch_size):
            all_distances += tflib.run(distance_expr)
        all_distances = np.concatenate(all_distances, axis=0)

        # Reject outliers.
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        self._report_result(np.mean(filtered_distances))

#----------------------------------------------------------------------------
