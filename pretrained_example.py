# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

import sys
import argparse


def get_args(arg_input):
    """Takes args input and returns them as a argparse parser

    Parameters
    -------------

    arg_input : list, shape (n_nargs,)
        contains list of arguments passed to function

    Returns
    -------------

    args : namespace
        contains namespace with keys and values for each parser argument

    """
    print(type(arg_input))
    parser = argparse.ArgumentParser(description='generate images from stylegan network')
    parser.add_argument(
        '-n',
        '--net_location',
        type=str,
        default=
        'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
        help="network file or url to load"
    )
    parser.add_argument(
        '--psi', type=float, default=0.7, help="truncation psi for generation"
    )
    parser.add_argument(
        '--num', type=int, default=1, help="number of images to generate"
    )
    parser.add_argument(
        '--random_seed', type=int, default=10, help="random seed"
    )

    args = parser.parse_args(arg_input)
    return args


def generate(net_location, psi, num, random_seed):
    """Takes args for network generation

    Parameters
    -------------

    net_location : str
        file or url of saved network
    
    psi: float
        generation variety control

    num: int
        number of images to generate

    random_seed: int
        random seed

    Returns
    -------------

    None

    """
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    if os.path.isfile(net_location):
        with open(net_location, 'rb') as f:
            _G, _D, Gs = pickle.load(f)
    else:
        with dnnlib.util.open_url(
            net_location, cache_dir=config.cache_dir
        ) as f:
            _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(random_seed)
    latents = rnd.randn(num, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(
        latents,
        None,
        truncation_psi=psi,
        randomize_noise=True,
        output_transform=fmt
    )

    # Save images.
    os.makedirs(config.result_dir, exist_ok=True)
    i = 0
    for image in images:
        png_filename = os.path.join(config.result_dir, f'example{i}.png')
        PIL.Image.fromarray(image, 'RGB').save(png_filename)
        i += 1


def main(args=None):
    if args == None:
        arg_input = sys.argv[1:]
        args = get_args(arg_input)

    generate(args.net_location, args.psi, args.num, args.random_seed)


if __name__ == "__main__":
    main()
