"""
perform the body tracking and create spatio temporal graph according to sliding window size
"""

import json
import sys
import os
from os import path
import torch
import numpy as np
from collections import deque
import time
import queue
from collections import deque

from action_detection.body_tracking.body_tracking_SDK import *
import action_detection.joints as jt


class SkeletonTracker(BodyTracker):
    """
    class for creating a spatio temporal graph out of data from azure kinect body tracking SDK (python wrapper)
    """

    def __init__(self):
        super().__init__()

        # # import settings
        # base_path = path.dirname(__file__)
        # file_path = path.abspath(path.join(base_path, "..", "..", "config_NN.json"))
        # with open(file_path) as json_file:
        #     settings = json.load(json_file)
        #
        # # sliding window size in frames
        # self.sw_size = settings["sliding_window_size"]

    def get_frame(self):
        """
        get one frame from azure kinect body tracking SDK
        :return:
        """
        sensor_capture = k4a.k4a_capture_t()
        get_capture_result = k4a.k4a_device_get_capture(self.device, ctypes.byref(sensor_capture),
                                                        k4a.K4A_WAIT_INFINITE)

        if get_capture_result == k4a.K4A_WAIT_RESULT_SUCCEEDED:

            queue_capture_result = k4a.k4abt_tracker_enqueue_capture(self.tracker, sensor_capture,
                                                                     k4a.K4A_WAIT_INFINITE)

            k4a.k4a_capture_release(sensor_capture)

            # error handling for timeout and failure of kinect
            if queue_capture_result == k4a.K4A_WAIT_RESULT_TIMEOUT:
                # It should never hit timeout when K4A_WAIT_INFINITE is set.
                raise Exception("Error! Add capture to tracker process queue timeout!")
                # print("Error! Add capture to tracker process queue timeout!")
                # return
            elif queue_capture_result == k4a.K4A_WAIT_RESULT_FAILED:
                raise Exception("Error! Add capture to tracker process queue timeout!")
                # print("Error! Add capture to tracker process queue failed!")
                # return

            # get frame
            body_frame = k4a.k4abt_frame_t()
            pop_frame_result = k4a.k4abt_tracker_pop_result(self.tracker, ctypes.byref(body_frame),
                                                            k4a.K4A_WAIT_INFINITE)
            if pop_frame_result == k4a.K4A_WAIT_RESULT_SUCCEEDED:
                num_bodies = k4a.k4abt_frame_get_num_bodies(body_frame)

                # only process first body (idx 0) ToDo: maybe extend later to multiple bodies
                i = 0
                body = k4a.k4abt_body_t()
                self.VERIFY(k4a.k4abt_frame_get_body_skeleton(body_frame, i, ctypes.byref(body.skeleton)),
                            "Get body from body frame failed!")
                body.id = k4a.k4abt_frame_get_body_id(body_frame, i)

                k4a.k4abt_frame_release(body_frame)
                # return body frame
                return body

            # doing some error handling:
            elif pop_frame_result == k4a.K4A_WAIT_RESULT_TIMEOUT:
                # It should never hit timeout when K4A_WAIT_INFINITE is set.
                raise Exception("Error! Pop body frame result timeout!")
                # print("Error! Pop body frame result timeout!")
                # return
            else:
                raise Exception("Pop body frame result failed!")
                # print("Pop body frame result failed!")
                # return
        elif get_capture_result == k4a.K4A_WAIT_RESULT_TIMEOUT:
            # It should never hit timeout when K4A_WAIT_INFINITE is set.
            raise Exception("Error! Get depth frame time out!")
            # print("Error! Get depth frame time out!")
            # return
        else:
            raise Exception("Get depth capture returned error: {}".format(get_capture_result))
            # print("Get depth capture returned error: {}".format(get_capture_result))


if __name__ == '__main__':
    g = SkeletonTracker()
    q = deque()
    num_frames = 120
    start = time.time()
    for i in range(num_frames):
        frame = g.get_frame()
        q.append(frame)
        # print(frame)
    end = time.time()
    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
    # Calculate frames per second
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    g.close_device()
