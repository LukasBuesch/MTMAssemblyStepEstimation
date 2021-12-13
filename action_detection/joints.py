import numpy as np
import torch
import torch.nn as nn

# Matrix defining the links between body joints
joint_links = np.array([[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]])

# list of joints used as index in InHARD dataset 
joint_used_InHARD = [0,  # hips - 0
                     7,  # spine - 1
                     11,  # neck - 2
                     12,  # head - 3
                     13,  # right shoulder - 4
                     14,  # right arm - 5
                     15,  # right forearm - 6
                     16,  # right hand - 7
                     36,  # left shoulder - 8
                     37,  # left arm - 9
                     38,  # left forearm - 10
                     39]  # left hand - 11



# list of joints used as index from Kinect
joint_used_Kinect = [0,  # hips - 0
                     1,  # spine - 1
                     3,  # neck - 2
                     27,  # head - 3
                     12,  # right shoulder - 4
                     13,  # right arm - 5 (elbow from kinect)
                     14,  # right forearm - 6 (wrist from kinect)
                     15,  # right hand - 7
                     5,  # left shoulder - 8
                     6,  # left arm - 9
                     7,  # left forearm - 10
                     8]  # left hand - 11

# Degrees of freedom per joint
joint_dof = 6  # 3 translations and 3 rotations

# reference joint for normalization
reference_joint = 1  # spine

# value for input normalization of coordinates
input_scaling_lower_bound = -0.95
input_scaling_upper_bound = 0.95

joints_dict = {
    0: "hips",
    1: "spine",
    2: "neck",
    3: "head",
    4: "right shoulder",
    5: "right arm",
    6: "right forearm",
    7: "right hand",
    8: "left shoulder",
    9: "left arm",
    10: "left forearm",
    11: "left hand",
}

if __name__ == '__main__':
    # print(joint_links)
    # print(joint_links.shape)
    # print(joints_dict[0])
    A = torch.tensor(joint_links, dtype=torch.float32, requires_grad=False)
    print(A.size())
    data_bn = nn.BatchNorm1d(6 * A.size(1))  # normalize
    a = 1
