import os
import math
import numpy as np
from numpy import savetxt
from bvh import Bvh
import action_detection.SW_GCN.action_labels as al
import torch
import torch.utils.data as data_utils
from datetime import *
import action_detection.joints as joints


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print("{} |{}| {}% {}".format(prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()


class TruncateExcept(Exception):
    pass


class LoadData:
    def __init__(self, w_size=60, is_train=True, squeeze_window=True):

        # window size for scaling
        self.w_size = w_size
        self.squeeze_window = squeeze_window
        self.left_out = 0

        self.is_train = is_train  # boolean for train (False for parsing validation data)

        # scaling all values from lower to upper bound for normalization input of network
        self.input_scaling_lower_bound = joints.input_scaling_lower_bound
        self.input_scaling_upper_bound = joints.input_scaling_upper_bound
        # path to data
        self.path_to_folder = "C:\\Users\\oculus\\Desktop\\PA_Buesch\\InHARD\\Segmented\\SkletonSegmented\\"

        # define training/validation data according to paper (InHARD)
        if self.is_train:
            self.data_index = ["P01_R01", "P01_R03", "P03_R01", "P03_R03", "P03_R04", "P04_R02", "P05_R03", "P05_R04",
                               "P06_R01", "P07_R01", "P07_R02", "P08_R02", "P08_R04", "P09_R01", "P09_R03", "P10_R01",
                               "P10_R02", "P10_R03", "P11_R02", "P12_R01", "P12_R02", "P13_R02", "P14_R01", "P15_R01",
                               "P15_R02", "P16_R02"]
            # self.data_index = ["P01_R01"]

        else:  # validation data
            self.data_index = ["P01_R02", "P02_R01", "P02_R02", "P04_R01", "P05_R01", "P05_R02",
                               "P08_R01", "P08_R03", "P09_R02", "P11_R01", "P14_R02", "P16_R01"]
            # self.data_index = ["P01_R02"]

        self.len_data_ind = len(self.data_index[0])

        # list of joints used as index in file (InHARD dataset)
        self.joint_used = joints.joint_used_InHARD

        self.reference_joint = joints.reference_joint  # spine

        self.num_pos = joints.joint_dof  # 3 translations and 3 rotations

        # variable to store training data in
        self.dataset_size = self.size_dataset_calc()
        self.data = np.empty((self.dataset_size, self.num_pos, self.w_size, len(self.joint_used)))
        self.label = np.empty(self.dataset_size)

    def parse_data(self):
        data_set_id = 0  # id of data set

        # load data:

        # iterate over folders
        for folder in os.listdir(self.path_to_folder):

            # skip actions not named in action labels
            if folder not in al.action_labels_train:
                continue

            # iterate over bvh files
            folder_path = self.path_to_folder + folder
            for filename in os.listdir(folder_path):
                if filename[0:self.len_data_ind] in self.data_index:
                    # read file
                    with open(folder_path + "\\" + filename) as f:
                        raw_data = Bvh(f.read())
                        self.cut_raw_data(raw_data)

                        # try to scale window
                        try:
                            self.scale_to_window(raw_data)
                        except TruncateExcept:
                            # when self.squeeze_window==False and num frames  in data shorter than sliding window don't
                            # use data
                            print("# frames in data < sliding window size - leave out data")
                            self.left_out += 1
                            continue

                        data = raw_data

                        # store data:

                        # add to train data:

                        # iterate over frames
                        counter_frames = 0
                        for frames in data.frames:
                            # iterate through frame
                            for element in range(len(frames)):
                                # train data = [dataset][coordinates][frame number][Joint number]
                                self.data[data_set_id][element % self.num_pos][counter_frames][
                                    math.floor(element / self.num_pos)] = frames[element]
                            counter_frames += 1
                        # store label for train data
                        # summarize similar action classes in same label
                        self.label[data_set_id] = al.get_action_class_from_action_labels_train(folder)

                        # loading progress
                        print_progress_bar(data_set_id, self.dataset_size,
                                           prefix='Progress train={} | w_size {} | squeeze_w {} :'.format(self.is_train,
                                                                                                          self.w_size,
                                                                                                          self.squeeze_window)
                                           , suffix='Complete | folder=\'{}\''.format(folder))
                        data_set_id += 1

                else:
                    continue
        print_progress_bar(data_set_id, self.dataset_size,
                           prefix='Progress train={} | w_size {} | squeeze_w {} :'.format(self.is_train, self.w_size,
                                                                                          self.squeeze_window),
                           suffix='Complete')

        # append one dimension at end
        self.data = np.expand_dims(self.data, len(self.data.shape))

        # convert strings to numbers
        self.data = np.array(self.data, dtype='float32')
        self.label = np.array(self.label, dtype='int64')

        # normalization and scaling:
        # normalize Joint positions to
        self.normalize_joint_pos()
        # scale to range [-1,1] -> better to process for NN
        self.scale_to_input_size(self.input_scaling_lower_bound, self.input_scaling_upper_bound)
        # create train data:
        train = data_utils.TensorDataset(torch.from_numpy(self.data),
                                         torch.from_numpy(self.label))

        if self.is_train:
            purpose = 'train'
        else:
            purpose = 'val'

        path = os.path.relpath

        torch.save(train, 'InHARD_{}_sw_{}_squ_{}_left_out_{}.pt'.format(purpose,
                                                                         self.w_size, self.squeeze_window,
                                                                         self.left_out))

    def scale_to_window(self, raw_data):
        """
        scale window to defined length (given in frames e.g. 60 frames ~ 0.5 sec)
        :param raw_data:
        :return:
        """
        if self.squeeze_window:
            """
            squeeze the data to fir in given sliding window size
            
            - leave out frames when data to long for sliding window
            - interpolate frames when data to short for sliding window
            """
            num_frames = len(raw_data.frames)
            # up scaling:
            if self.w_size > num_frames:
                # multiple every frame
                s = math.ceil(self.w_size / num_frames)  # scaling factor rounded up
                for i in range(num_frames * s):
                    interpolated_frames = self.interpolate_frames(raw_data.frames[i], raw_data.frames[i + 1], s)
                    for j in range(s - 1):
                        raw_data.frames.insert(i + j, interpolated_frames[j])
                    i += (s - 1)

            # down scaling / squeezing:
            if self.w_size < num_frames:
                s = math.floor(num_frames / self.w_size)  # rounds down to integer value

                # delete conservative to not receive length shorter predefined
                # iterate backwards
                for i in range(num_frames):
                    # keep every s pos
                    if (num_frames - 1 - i) % s != 0:
                        del raw_data.frames[num_frames - 1 - i]

            # cut end to receive length predefined
            for i in range(self.w_size, len(raw_data.frames)):
                if i % 2 == 0:
                    del raw_data.frames[0]
                else:
                    del raw_data.frames[self.w_size]

        else:
            """
            - truncate data when to long for sliding window
            - leave out data when to short for sliding window
            - if no squeeze the fps of dataset (InHARD - 120 Hz) and 
              camera application (Kinect python wrapper = 24 Hz, but in multithreading application = 9 Hz)
              -> 120/9 = 13 = 7*2
            """

            # delete frames to get to low frame rate
            while len(raw_data.frames) > 2 * self.w_size:
                num_frames = len(raw_data.frames)
                for i in range(len(raw_data.frames)):
                    if i % 2 == 0:
                        del raw_data.frames[(num_frames-1) - i]

            # leave out:
            if self.w_size > len(raw_data.frames):
                raise TruncateExcept()

            # truncate: cut end to receive length predefined
            for i in range(self.w_size, len(raw_data.frames)):
                if i % 2 == 0:
                    del raw_data.frames[0]
                else:
                    del raw_data.frames[self.w_size]

    def cut_raw_data(self, raw_data):
        """
        remove unused joints
        :return:
        """

        # iterate over frames
        frame_length = len(raw_data.frames[0])
        for frame in raw_data.frames:

            # iterate backwards
            for i in range(frame_length):
                # delete joints not used later
                if not self.is_used_joint(frame_length - 1 - i):
                    del frame[frame_length - 1 - i]

    def is_used_joint(self, i):
        """
        check if joint used for data
        :param i:
        :return:
        """
        # map frame position to body joint index
        for joint in self.joint_used:
            if joint * self.num_pos <= i < (joint + 1) * self.num_pos:
                return True
        return False

    def size_dataset_calc(self):
        counter = 0
        for folder in os.listdir(self.path_to_folder):

            # skip actions not named in action labels
            if folder not in al.action_labels_train:
                continue

            # iterate over bvh files
            folder_path = self.path_to_folder + folder
            for filename in os.listdir(folder_path):
                if filename[0:self.len_data_ind] in self.data_index:
                    counter += 1
        return counter

    def interpolate_frames(self, prev_frame, next_frame, scaling):
        """
        interpolate frame between two following frames
        :param prev_frame:
        :param next_frame:
        :param scaling:
        :return:
        """
        interpolated_frames = []
        # number of frames to generate
        for i in range(scaling - 1):
            frame = []
            # every frame element
            for j in range(len(prev_frame)):
                frame.append(str(float(prev_frame[j]) + (float(next_frame[j]) - float(prev_frame[j]))
                                 * (i / scaling)))  # linear interpolation
            interpolated_frames.append(frame)
        return interpolated_frames

    def normalize_joint_pos(self):
        """
        normalize joint positions to spine joint to attenuate effects of inertia drifts
        :return:
        """
        for dataset in range(len(self.data)):
            for coordinates in range(len(self.data[dataset])):
                for frame in range(len(self.data[dataset][coordinates])):
                    for joint in range(len(self.data[dataset][coordinates][frame])):
                        self.data[dataset][coordinates][frame][joint] = \
                            self.data[dataset][coordinates][frame][joint] \
                            - self.data[dataset][coordinates][frame][self.reference_joint]

    def scale_to_input_size(self, lower_value, upper_value):
        """
        first calc to max and min of every of the six coordinates and then scale to given interval
        :param lower_value:
        :param upper_value:
        :return:
        """
        maxima = [0, 0, 0, 0, 0, 0]
        minima = [0, 0, 0, 0, 0, 0]
        # search for max and min
        for dataset in range(len(self.data)):
            for coordinates in range(len(self.data[dataset])):
                for frame in range(len(self.data[dataset][coordinates])):
                    for joint in range(len(self.data[dataset][coordinates][frame])):
                        if self.data[dataset][coordinates][frame][joint] > maxima[coordinates]:
                            maxima[coordinates] = self.data[dataset][coordinates][frame][joint]
                        if self.data[dataset][coordinates][frame][joint] < minima[coordinates]:
                            minima[coordinates] = self.data[dataset][coordinates][frame][joint]

        # scale according the precalculated max and min and interval given
        for dataset in range(len(self.data)):
            for coordinates in range(len(self.data[dataset])):
                for frame in range(len(self.data[dataset][coordinates])):
                    for joint in range(len(self.data[dataset][coordinates][frame])):
                        self.data[dataset][coordinates][frame][joint] = \
                            (self.data[dataset][coordinates][frame][joint] - minima[coordinates]) \
                            * (upper_value - lower_value) / (maxima[coordinates] - minima[coordinates]) \
                            + lower_value


if __name__ == '__main__':
    for squeeze in [True, False]:

        window_sizes = [40, 30, 20, 10, 160, 120, 100, 80, 60, 50, 5]
        # window_sizes = [5]
        for size in window_sizes:
            # training data
            instance_train = LoadData(size, is_train=True, squeeze_window=squeeze)
            instance_train.parse_data()
            del instance_train

            # validation data
            instance_val = LoadData(size, is_train=False, squeeze_window=squeeze)
            instance_val.parse_data()
            del instance_val
