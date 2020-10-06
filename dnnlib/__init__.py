# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from dnnlib import submission

from dnnlib.submission.run_context import RunContext

from dnnlib.submission.submit import SubmitTarget
from dnnlib.submission.submit import PathType
from dnnlib.submission.submit import SubmitConfig
from dnnlib.submission.submit import get_path_from_template
from dnnlib.submission.submit import submit_run

from dnnlib.util import EasyDict

submit_config: SubmitConfig = None # Package level variable for SubmitConfig which is only valid when inside the run function.
