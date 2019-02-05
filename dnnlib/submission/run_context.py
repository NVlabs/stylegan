# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Helpers for managing the run/training loop."""

import datetime
import json
import os
import pprint
import time
import types

from typing import Any

from . import submit


class RunContext(object):
    """Helper class for managing the run/training loop.

    The context will hide the implementation details of a basic run/training loop.
    It will set things up properly, tell if run should be stopped, and then cleans up.
    User should call update periodically and use should_stop to determine if run should be stopped.

    Args:
        submit_config: The SubmitConfig that is used for the current run.
        config_module: The whole config module that is used for the current run.
        max_epoch: Optional cached value for the max_epoch variable used in update.
    """

    def __init__(self, submit_config: submit.SubmitConfig, config_module: types.ModuleType = None, max_epoch: Any = None):
        self.submit_config = submit_config
        self.should_stop_flag = False
        self.has_closed = False
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_update_interval = 0.0
        self.max_epoch = max_epoch

        # pretty print the all the relevant content of the config module to a text file
        if config_module is not None:
            with open(os.path.join(submit_config.run_dir, "config.txt"), "w") as f:
                filtered_dict = {k: v for k, v in config_module.__dict__.items() if not k.startswith("_") and not isinstance(v, (types.ModuleType, types.FunctionType, types.LambdaType, submit.SubmitConfig, type))}
                pprint.pprint(filtered_dict, stream=f, indent=4, width=200, compact=False)

        # write out details about the run to a text file
        self.run_txt_data = {"task_name": submit_config.task_name, "host_name": submit_config.host_name, "start_time": datetime.datetime.now().isoformat(sep=" ")}
        with open(os.path.join(submit_config.run_dir, "run.txt"), "w") as f:
            pprint.pprint(self.run_txt_data, stream=f, indent=4, width=200, compact=False)

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def update(self, loss: Any = 0, cur_epoch: Any = 0, max_epoch: Any = None) -> None:
        """Do general housekeeping and keep the state of the context up-to-date.
        Should be called often enough but not in a tight loop."""
        assert not self.has_closed

        self.last_update_interval = time.time() - self.last_update_time
        self.last_update_time = time.time()

        if os.path.exists(os.path.join(self.submit_config.run_dir, "abort.txt")):
            self.should_stop_flag = True

        max_epoch_val = self.max_epoch if max_epoch is None else max_epoch

    def should_stop(self) -> bool:
        """Tell whether a stopping condition has been triggered one way or another."""
        return self.should_stop_flag

    def get_time_since_start(self) -> float:
        """How much time has passed since the creation of the context."""
        return time.time() - self.start_time

    def get_time_since_last_update(self) -> float:
        """How much time has passed since the last call to update."""
        return time.time() - self.last_update_time

    def get_last_update_interval(self) -> float:
        """How much time passed between the previous two calls to update."""
        return self.last_update_interval

    def close(self) -> None:
        """Close the context and clean up.
        Should only be called once."""
        if not self.has_closed:
            # update the run.txt with stopping time
            self.run_txt_data["stop_time"] = datetime.datetime.now().isoformat(sep=" ")
            with open(os.path.join(self.submit_config.run_dir, "run.txt"), "w") as f:
                pprint.pprint(self.run_txt_data, stream=f, indent=4, width=200, compact=False)

            self.has_closed = True
