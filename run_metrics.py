# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main entry point for training StyleGAN and ProGAN networks."""

import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib

import config
from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

def run_pickle(submit_config, metric_args, network_pkl, dataset_args, mirror_augment):
    ctx = dnnlib.RunContext(submit_config)
    tflib.init_tf()
    print('Evaluating %s metric on network_pkl "%s"...' % (metric_args.name, network_pkl))
    metric = dnnlib.util.call_func_by_name(**metric_args)
    print()
    metric.run(network_pkl, dataset_args=dataset_args, mirror_augment=mirror_augment, num_gpus=submit_config.num_gpus)
    print()
    ctx.close()

#----------------------------------------------------------------------------

def run_snapshot(submit_config, metric_args, run_id, snapshot):
    ctx = dnnlib.RunContext(submit_config)
    tflib.init_tf()
    print('Evaluating %s metric on run_id %s, snapshot %s...' % (metric_args.name, run_id, snapshot))
    run_dir = misc.locate_run_dir(run_id)
    network_pkl = misc.locate_network_pkl(run_dir, snapshot)
    metric = dnnlib.util.call_func_by_name(**metric_args)
    print()
    metric.run(network_pkl, run_dir=run_dir, num_gpus=submit_config.num_gpus)
    print()
    ctx.close()

#----------------------------------------------------------------------------

def run_all_snapshots(submit_config, metric_args, run_id):
    ctx = dnnlib.RunContext(submit_config)
    tflib.init_tf()
    print('Evaluating %s metric on all snapshots of run_id %s...' % (metric_args.name, run_id))
    run_dir = misc.locate_run_dir(run_id)
    network_pkls = misc.list_network_pkls(run_dir)
    metric = dnnlib.util.call_func_by_name(**metric_args)
    print()
    for idx, network_pkl in enumerate(network_pkls):
        ctx.update('', idx, len(network_pkls))
        metric.run(network_pkl, run_dir=run_dir, num_gpus=submit_config.num_gpus)
    print()
    ctx.close()

#----------------------------------------------------------------------------

def main():
    # Which metrics to evaluate?
    metrics = []
    metrics += [metric_base.fid50k]
    #metrics += [metric_base.ppl_zfull]
    #metrics += [metric_base.ppl_wfull]
    #metrics += [metric_base.ppl_zend]
    #metrics += [metric_base.ppl_wend]
    #metrics += [metric_base.ls]
    #metrics += [metric_base.dummy]

    # Which networks to evaluate them on?
    tasks = []
    tasks += [EasyDict(run_func_name='run_metrics.run_pickle', network_pkl='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', dataset_args=EasyDict(tfrecord_dir='ffhq', shuffle_mb=0), mirror_augment=True)] # karras2019stylegan-ffhq-1024x1024.pkl
    #tasks += [EasyDict(run_func_name='run_metrics.run_snapshot', run_id=100, snapshot=25000)]
    #tasks += [EasyDict(run_func_name='run_metrics.run_all_snapshots', run_id=100)]

    # How many GPUs to use?
    submit_config.num_gpus = 1
    #submit_config.num_gpus = 2
    #submit_config.num_gpus = 4
    #submit_config.num_gpus = 8

    # Execute.
    submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    for task in tasks:
        for metric in metrics:
            submit_config.run_desc = '%s-%s' % (task.run_func_name, metric.name)
            if task.run_func_name.endswith('run_snapshot'):
                submit_config.run_desc += '-%s-%s' % (task.run_id, task.snapshot)
            if task.run_func_name.endswith('run_all_snapshots'):
                submit_config.run_desc += '-%s' % task.run_id
            submit_config.run_desc += '-%dgpu' % submit_config.num_gpus
            dnnlib.submit_run(submit_config, metric_args=metric, **task)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
