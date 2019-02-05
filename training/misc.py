# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Miscellaneous utility functions."""

import os
import glob
import pickle
import re
import numpy as np
from collections import defaultdict
import PIL.Image
import dnnlib

import config
from training import dataset

#----------------------------------------------------------------------------
# Convenience wrappers for pickle that are able to load data produced by
# older versions of the code, and from external URLs.

def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir=config.cache_dir)
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------------
# Image utils.

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)

def save_image(image, filename, drange=[0,1], quality=95):
    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)

def save_image_grid(images, filename, drange=[0,1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)

#----------------------------------------------------------------------------
# Locating results.

def locate_run_dir(run_id_or_run_dir):
    if isinstance(run_id_or_run_dir, str):
        if os.path.isdir(run_id_or_run_dir):
            return run_id_or_run_dir
        converted = dnnlib.submission.submit.convert_path(run_id_or_run_dir)
        if os.path.isdir(converted):
            return converted

    run_dir_pattern = re.compile('^0*%s-' % str(run_id_or_run_dir))
    for search_dir in ['']:
        full_search_dir = config.result_dir if search_dir == '' else os.path.normpath(os.path.join(config.result_dir, search_dir))
        run_dir = os.path.join(full_search_dir, str(run_id_or_run_dir))
        if os.path.isdir(run_dir):
            return run_dir
        run_dirs = sorted(glob.glob(os.path.join(full_search_dir, '*')))
        run_dirs = [run_dir for run_dir in run_dirs if run_dir_pattern.match(os.path.basename(run_dir))]
        run_dirs = [run_dir for run_dir in run_dirs if os.path.isdir(run_dir)]
        if len(run_dirs) == 1:
            return run_dirs[0]
    raise IOError('Cannot locate result subdir for run', run_id_or_run_dir)

def list_network_pkls(run_id_or_run_dir, include_final=True):
    run_dir = locate_run_dir(run_id_or_run_dir)
    pkls = sorted(glob.glob(os.path.join(run_dir, 'network-*.pkl')))
    if len(pkls) >= 1 and os.path.basename(pkls[0]) == 'network-final.pkl':
        if include_final:
            pkls.append(pkls[0])
        del pkls[0]
    return pkls

def locate_network_pkl(run_id_or_run_dir_or_network_pkl, snapshot_or_network_pkl=None):
    for candidate in [snapshot_or_network_pkl, run_id_or_run_dir_or_network_pkl]:
        if isinstance(candidate, str):
            if os.path.isfile(candidate):
                return candidate
            converted = dnnlib.submission.submit.convert_path(candidate)
            if os.path.isfile(converted):
                return converted

    pkls = list_network_pkls(run_id_or_run_dir_or_network_pkl)
    if len(pkls) >= 1 and snapshot_or_network_pkl is None:
        return pkls[-1]

    for pkl in pkls:
        try:
            name = os.path.splitext(os.path.basename(pkl))[0]
            number = int(name.split('-')[-1])
            if number == snapshot_or_network_pkl:
                return pkl
        except ValueError: pass
        except IndexError: pass
    raise IOError('Cannot locate network pkl for snapshot', snapshot_or_network_pkl)

def get_id_string_for_network_pkl(network_pkl):
    p = network_pkl.replace('.pkl', '').replace('\\', '/').split('/')
    return '-'.join(p[max(len(p) - 2, 0):])

#----------------------------------------------------------------------------
# Loading data from previous training runs.

def load_network_pkl(run_id_or_run_dir_or_network_pkl, snapshot_or_network_pkl=None):
    return load_pkl(locate_network_pkl(run_id_or_run_dir_or_network_pkl, snapshot_or_network_pkl))

def parse_config_for_previous_run(run_id):
    run_dir = locate_run_dir(run_id)

    # Parse config.txt.
    cfg = defaultdict(dict)
    with open(os.path.join(run_dir, 'config.txt'), 'rt') as f:
        for line in f:
            line = re.sub(r"^{?\s*'(\w+)':\s*{(.*)(},|}})$", r"\1 = {\2}", line.strip())
            if line.startswith('dataset =') or line.startswith('train ='):
                exec(line, cfg, cfg) # pylint: disable=exec-used

    # Handle legacy options.
    if 'file_pattern' in cfg['dataset']:
        cfg['dataset']['tfrecord_dir'] = cfg['dataset'].pop('file_pattern').replace('-r??.tfrecords', '')
    if 'mirror_augment' in cfg['dataset']:
        cfg['train']['mirror_augment'] = cfg['dataset'].pop('mirror_augment')
    if 'max_labels' in cfg['dataset']:
        v = cfg['dataset'].pop('max_labels')
        if v is None: v = 0
        if v == 'all': v = 'full'
        cfg['dataset']['max_label_size'] = v
    if 'max_images' in cfg['dataset']:
        cfg['dataset'].pop('max_images')
    return cfg

def load_dataset_for_previous_run(run_id, **kwargs): # => dataset_obj, mirror_augment
    cfg = parse_config_for_previous_run(run_id)
    cfg['dataset'].update(kwargs)
    dataset_obj = dataset.load_dataset(data_dir=config.data_dir, **cfg['dataset'])
    mirror_augment = cfg['train'].get('mirror_augment', False)
    return dataset_obj, mirror_augment

def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch

#----------------------------------------------------------------------------
# Size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(G, training_set,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    gw = 1; gh = 1
    if size == '1080p':
        gw = np.clip(1920 // G.output_shape[3], 3, 32)
        gh = np.clip(1080 // G.output_shape[2], 2, 32)
    if size == '4k':
        gw = np.clip(3840 // G.output_shape[3], 7, 32)
        gh = np.clip(2160 // G.output_shape[2], 4, 32)

    # Initialize data arrays.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    latents = np.random.randn(gw * gh, *G.input_shape[1:])

    # Random layout.
    if layout == 'random':
        reals[:], labels[:] = training_set.get_minibatch_np(gw * gh)

    # Class-conditional layouts.
    class_layouts = dict(row_per_class=[gw,1], col_per_class=[1,gh], class4x4=[4,4])
    if layout in class_layouts:
        bw, bh = class_layouts[layout]
        nw = (gw - 1) // bw + 1
        nh = (gh - 1) // bh + 1
        blocks = [[] for _i in range(nw * nh)]
        for _iter in range(1000000):
            real, label = training_set.get_minibatch_np(1)
            idx = np.argmax(label[0])
            while idx < len(blocks) and len(blocks[idx]) >= bw * bh:
                idx += training_set.label_size
            if idx < len(blocks):
                blocks[idx].append((real, label))
                if all(len(block) >= bw * bh for block in blocks):
                    break
        for i, block in enumerate(blocks):
            for j, (real, label) in enumerate(block):
                x = (i %  nw) * bw + j %  bw
                y = (i // nw) * bh + j // bw
                if x < gw and y < gh:
                    reals[x + y * gw] = real[0]
                    labels[x + y * gw] = label[0]

    return (gw, gh), reals, labels, latents

#----------------------------------------------------------------------------
