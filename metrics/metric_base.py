# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Common definitions for GAN metrics."""

import os
import time
import hashlib
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

import config
from training import misc
from training import dataset

#----------------------------------------------------------------------------
# Standard metrics.

fid50k = dnnlib.EasyDict(func_name='metrics.frechet_inception_distance.FID', name='fid50k', num_images=50000, minibatch_per_gpu=8)
ppl_zfull = dnnlib.EasyDict(func_name='metrics.perceptual_path_length.PPL', name='ppl_zfull', num_samples=100000, epsilon=1e-4, space='z', sampling='full', minibatch_per_gpu=16)
ppl_wfull = dnnlib.EasyDict(func_name='metrics.perceptual_path_length.PPL', name='ppl_wfull', num_samples=100000, epsilon=1e-4, space='w', sampling='full', minibatch_per_gpu=16)
ppl_zend = dnnlib.EasyDict(func_name='metrics.perceptual_path_length.PPL', name='ppl_zend', num_samples=100000, epsilon=1e-4, space='z', sampling='end', minibatch_per_gpu=16)
ppl_wend = dnnlib.EasyDict(func_name='metrics.perceptual_path_length.PPL', name='ppl_wend', num_samples=100000, epsilon=1e-4, space='w', sampling='end', minibatch_per_gpu=16)
ls = dnnlib.EasyDict(func_name='metrics.linear_separability.LS', name='ls', num_samples=200000, num_keep=100000, attrib_indices=range(40), minibatch_per_gpu=4)
dummy = dnnlib.EasyDict(func_name='metrics.metric_base.DummyMetric', name='dummy') # for debugging

#----------------------------------------------------------------------------
# Base class for metrics.

class MetricBase:
    def __init__(self, name):
        self.name = name
        self._network_pkl = None
        self._dataset_args = None
        self._mirror_augment = None
        self._results = []
        self._eval_time = None

    def run(self, network_pkl, run_dir=None, dataset_args=None, mirror_augment=None, num_gpus=1, tf_config=None, log_results=True):
        self._network_pkl = network_pkl
        self._dataset_args = dataset_args
        self._mirror_augment = mirror_augment
        self._results = []

        if (dataset_args is None or mirror_augment is None) and run_dir is not None:
            run_config = misc.parse_config_for_previous_run(run_dir)
            self._dataset_args = dict(run_config['dataset'])
            self._dataset_args['shuffle_mb'] = 0
            self._mirror_augment = run_config['train'].get('mirror_augment', False)

        time_begin = time.time()
        with tf.Graph().as_default(), tflib.create_session(tf_config).as_default(): # pylint: disable=not-context-manager
            _G, _D, Gs = misc.load_pkl(self._network_pkl)
            self._evaluate(Gs, num_gpus=num_gpus)
        self._eval_time = time.time() - time_begin

        if log_results:
            result_str = self.get_result_str()
            if run_dir is not None:
                log = os.path.join(run_dir, 'metric-%s.txt' % self.name)
                with dnnlib.util.Logger(log, 'a'):
                    print(result_str)
            else:
                print(result_str)

    def get_result_str(self):
        network_name = os.path.splitext(os.path.basename(self._network_pkl))[0]
        if len(network_name) > 29:
            network_name = '...' + network_name[-26:]
        result_str = '%-30s' % network_name
        result_str += ' time %-12s' % dnnlib.util.format_time(self._eval_time)
        for res in self._results:
            result_str += ' ' + self.name + res.suffix + ' '
            result_str += res.fmt % res.value
        return result_str

    def update_autosummaries(self):
        for res in self._results:
            tflib.autosummary.autosummary('Metrics/' + self.name + res.suffix, res.value)

    def _evaluate(self, Gs, num_gpus):
        raise NotImplementedError # to be overridden by subclasses

    def _report_result(self, value, suffix='', fmt='%-10.4f'):
        self._results += [dnnlib.EasyDict(value=value, suffix=suffix, fmt=fmt)]

    def _get_cache_file_for_reals(self, extension='pkl', **kwargs):
        all_args = dnnlib.EasyDict(metric_name=self.name, mirror_augment=self._mirror_augment)
        all_args.update(self._dataset_args)
        all_args.update(kwargs)
        md5 = hashlib.md5(repr(sorted(all_args.items())).encode('utf-8'))
        dataset_name = self._dataset_args['tfrecord_dir'].replace('\\', '/').split('/')[-1]
        return os.path.join(config.cache_dir, '%s-%s-%s.%s' % (md5.hexdigest(), self.name, dataset_name, extension))

    def _iterate_reals(self, minibatch_size):
        dataset_obj = dataset.load_dataset(data_dir=config.data_dir, **self._dataset_args)
        while True:
            images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
            if self._mirror_augment:
                images = misc.apply_mirror_augment(images)
            yield images

    def _iterate_fakes(self, Gs, minibatch_size, num_gpus):
        while True:
            latents = np.random.randn(minibatch_size, *Gs.input_shape[1:])
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, output_transform=fmt, is_validation=True, num_gpus=num_gpus, assume_frozen=True)
            yield images

#----------------------------------------------------------------------------
# Group of multiple metrics.

class MetricGroup:
    def __init__(self, metric_kwarg_list):
        self.metrics = [dnnlib.util.call_func_by_name(**kwargs) for kwargs in metric_kwarg_list]

    def run(self, *args, **kwargs):
        for metric in self.metrics:
            metric.run(*args, **kwargs)

    def get_result_str(self):
        return ' '.join(metric.get_result_str() for metric in self.metrics)

    def update_autosummaries(self):
        for metric in self.metrics:
            metric.update_autosummaries()

#----------------------------------------------------------------------------
# Dummy metric for debugging purposes.

class DummyMetric(MetricBase):
    def _evaluate(self, Gs, num_gpus):
        _ = Gs, num_gpus
        self._report_result(0.0)

#----------------------------------------------------------------------------
