#!/usr/bin/env python

from distutils.core import setup

setup(name='stylegan',
      version='0.1',
      description='StyleGAN — Official TensorFlow Implementation',
      author='Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)',
      author_email='researchinquiries@nvidia.com',
      url='https://github.com/NVlabs/stylegan',
      packages=['dnnlib', 'metrics', 'training'],
     )
