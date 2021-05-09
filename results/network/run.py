# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Helper for launching run functions in computing clusters.

During the submit process, this file is copied to the appropriate run dir.
When the job is launched in the cluster, this module is the first thing that
is run inside the docker container.
"""

import os
import pickle
import sys

# PYTHONPATH should have been set so that the run_dir/src is in it
import dnnlib

def main():
    if not len(sys.argv) >= 4:
        raise RuntimeError("This script needs three arguments: run_dir, task_name and host_name!")

    run_dir = str(sys.argv[1])
    task_name = str(sys.argv[2])
    host_name = str(sys.argv[3])

    submit_config_path = os.path.join(run_dir, "submit_config.pkl")

    # SubmitConfig should have been pickled to the run dir
    if not os.path.exists(submit_config_path):
        raise RuntimeError("SubmitConfig pickle file does not exist!")

    submit_config: dnnlib.SubmitConfig = pickle.load(open(submit_config_path, "rb"))
    dnnlib.submission.submit.set_user_name_override(submit_config.user_name)

    submit_config.task_name = task_name
    submit_config.host_name = host_name

    dnnlib.submission.submit.run_wrapper(submit_config)

if __name__ == "__main__":
    main()
