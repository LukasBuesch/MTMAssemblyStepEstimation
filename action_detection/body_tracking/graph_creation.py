import numpy as np
import torch
import json
from os import path

import action_detection.joints as jt


class GraphCreator:

    def __init__(self, sw_sizes, device):

        # sliding window size for different networks
        self.sw_sizes = sw_sizes

        self.device = device

    def get_st_graph(self, frames):
        """
        method to call from main system to get the current spatio temporal graph
        :return:
        """
        # ToDo: for performance reasons don't calculate multiple times since parts of graph are same
        st_graphs = []
        confidences = []
        for graph in range(len(self.sw_sizes)):
            # capture number of frames according to sliding window size
            data = np.empty((jt.joint_dof, self.sw_sizes[graph], len(jt.joint_used_Kinect)))
            # frames = []
            confidence_per_frame = []
            # take the most recent frames as scope for sliding window
            for i in range(self.sw_sizes[graph]):
                data, confidence = self.transform_raw_frame(data, i, frames[len(frames) - self.sw_sizes[graph] + i])

                confidence_per_frame.append(confidence)

            # convert type to fit to training data
            data = np.array(data, dtype='float32')

            # normalize data to reference joint
            data = self.normalize_joint_positions(data)

            # # scale to range [-1,1] -> better to process for NN
            data = self.scale_to_input_size(data, jt.input_scaling_lower_bound, jt.input_scaling_upper_bound)

            # append one dimension at the end
            data = np.expand_dims(data, len(data.shape))

            # convert to torch tensor
            data = torch.from_numpy(data)

            # push data to gpu
            data = data.to(self.device)

            st_graphs.append(data)
            confidences.append(confidence_per_frame)

        return st_graphs, confidences

    def transform_raw_frame(self, frames, frame_idx, raw_frame):
        """
        - remove unused joints
        - build an average confidence level for whole frame over all joints used
        :return:
        """

        confidence_avg = 0
        confidence_count = 0
        # iterate over joints used from Kinect
        for joint_idx in range(len(jt.joint_used_Kinect)):

            # store position and orientation into frames array:
            # 3 translational coordinates
            for i in range(0, 3):
                frames[i][frame_idx][joint_idx] = \
                    raw_frame.skeleton.joints[jt.joint_used_Kinect[joint_idx]].position.v[i]
            # 3 rotational coordinates
            for i in range(0, 3):
                frames[i + 3][frame_idx][joint_idx] = \
                    raw_frame.skeleton.joints[jt.joint_used_Kinect[joint_idx]].orientation.v[i]

            # update confidence
            confidence_avg += raw_frame.skeleton.joints[jt.joint_used_Kinect[joint_idx]].confidence_level
            confidence_count += 1
            # append result to frames list

        # build average of cumulated confidence
        confidence_avg = round(confidence_avg / confidence_count)
        return frames, confidence_avg

    def scale_to_input_size(self, data, lower_value, upper_value):
        """
        first calc to max and min of every of the six coordinates and then scale to given interval
        :param data:
        :param lower_value:
        :param upper_value:
        :return:
        """
        maxima = [0, 0, 0, 0, 0, 0]
        minima = [0, 0, 0, 0, 0, 0]
        # search for max and min

        for coordinates in range(len(data)):
            for frame in range(len(data[coordinates])):
                for joint in range(len(data[coordinates][frame])):
                    if data[coordinates][frame][joint] > maxima[coordinates]:
                        maxima[coordinates] = data[coordinates][frame][joint]
                    if data[coordinates][frame][joint] < minima[coordinates]:
                        minima[coordinates] = data[coordinates][frame][joint]

        # scale according the precalculated max and min and interval given

        for coordinates in range(len(data)):
            for frame in range(len(data[coordinates])):
                for joint in range(len(data[coordinates][frame])):
                    try:
                        data[coordinates][frame][joint] = \
                            (data[coordinates][frame][joint] - minima[coordinates]) \
                            * (upper_value - lower_value) / (maxima[coordinates] - minima[coordinates]) \
                            + lower_value
                    except ZeroDivisionError:
                        data[coordinates][frame][joint] = \
                            (data[coordinates][frame][joint] - minima[coordinates])

        return data

    def normalize_joint_positions(self, data):
        """

        :return:
        """
        for coordinates in range(len(data)):
            for frame in range(len(data[coordinates])):
                for joint in range(len(data[coordinates][frame])):
                    data[coordinates][frame][joint] = \
                        data[coordinates][frame][joint] \
                        - data[coordinates][frame][jt.reference_joint]
        return data
