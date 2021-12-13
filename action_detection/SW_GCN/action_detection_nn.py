"""
instance for trained st_gcn network to use in action detection
"""

import json
import numpy as np

from action_detection.SW_GCN.network_model import *
from action_detection.SW_GCN.action_labels import action_names


class ActonDetectionNN:
    def __init__(self, sw_size, device):
        # import settings
        with open('config_NN.json') as json_file:
            settings = json.load(json_file)

        # set gpu device
        self.gpu_device = device

        classes_names = action_names

        # load model and push to gpu
        # torch.cuda.empty_cache()
        self.model = Model(settings["num_coord_per_joint"], len(classes_names), classes_names, self.gpu_device)
        # load existing weights for certain sliding window length
        print("load existing weights: {}".format(settings["model_for_sw_" + str(sw_size)]))
        self.model.load_state_dict(torch.load(settings["model_for_sw_" + str(sw_size)]))
        self.model = self.model.cuda()
        # set model to evaluation mode
        self.model.eval()

    def run_network(self, st_graph):
        """
        eun model for st graph
        :param st_graph:
        :return:
        """
        # push graph to cuda in graph creation thread for performance reasons
        # st_graph = st_graph.to(self.gpu_device)
        action_primitive = self.model(st_graph)

        # convert action primitive list to integer value of detected action class
        return action_primitive


if __name__ == '__main__':
    a = ActonDetectionNN(10, start_cuda())

