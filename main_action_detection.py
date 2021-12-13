import threading
from collections import deque
import queue
import torch
import time

# import modules
from action_detection.body_tracking.body_tracking import SkeletonTracker
from action_detection.body_tracking.graph_creation import GraphCreator
from action_detection.SW_GCN.action_detection_nn import ActonDetectionNN
from assembly_plan.MTM_compare.mtm_compare import MTMCompare

import action_detection.SW_GCN.action_labels as al
from action_detection.SW_GCN.network_model import start_cuda

"""
create and manage threads 
"""


class BodyTrackingThread(threading.Thread):
    """
    Thread for body tracking

    - communicates with STGraphThread via queue_skeleton

    """

    def __init__(self, thread_id, queue_skeleton, sw_sizes):
        threading.Thread.__init__(self)
        self.queue_skeleton = queue_skeleton
        self.threadID = thread_id
        self.sw_sizes = sw_sizes

        self.body_tracker = SkeletonTracker()

        self.is_stop = False

    def run(self):
        """
       Do cool stuff for body tracking here
       :return:
       """
        while not self.is_stop:
            # print("Run thread:{}".format(self.threadID))
            # get frame from azure kinect body tracking SDK
            # delete old items when max sw size is reached
            frame = self.body_tracker.get_frame()

            # push current frame to queue
            self.queue_skeleton.append(frame)

        # close azure kinect device when stopping thread
        self.body_tracker.close_device()

    def stop(self):
        self.is_stop = True


class STGraphThread(threading.Thread):
    """
    Thread for creating the spatio temporal graphs by applying the sliding window on camera frames from
    BodyTrackingThread

    - communicates with BodyTrackingThread via queue_skeleton
    - communicates with NeuralNetworkThread(s) vie queue_graph
    """

    def __init__(self, thread_id, queue_skeleton, queues_graph, sw_sizes, device):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.queue_skeleton = queue_skeleton
        self.queues_graph = queues_graph
        self.sw_sizes = sw_sizes
        self.device = device

        self.graph_creator = GraphCreator(self.sw_sizes, device)

        self.is_stop = False

    def run(self):
        """
       Do cool stuff for graph creation here
       :return:
       """
        fps_flag = False
        # check to not process on same data several times (no join operator for deque)
        frames_old = [1]
        # print("Run thread:{}".format(self.threadID))
        while not self.is_stop:
            # measure fps of camera
            # print(len(self.queue_skeleton))
            if len(self.queue_skeleton) == 1:
                start = time.time()
            if len(self.queue_skeleton) == max(self.sw_sizes) and not fps_flag:
                stop = time.time()
                seconds = stop - start
                # Calculate frames per second
                fps = max(self.sw_sizes) / seconds
                print("Estimated frames per second : {0}".format(fps))
                fps_flag = True

            # wait until queue has enough information for longest sliding window
            if len(self.queue_skeleton) > max(self.sw_sizes):

                # delete old items when max sw size is reached
                self.queue_skeleton.popleft()

                # get frame from body tracker without removing items from queue
                frames = list(self.queue_skeleton.copy())
                # print(frames[0])

                if frames_old[0] == frames[0]:
                    continue
                # print("New frame for graph")
                frames_old = frames

                # create graphs out of frames by applying sliding windows
                # ToDo: maybe use confidence of body tracking SDK later
                st_graphs, _ = self.graph_creator.get_st_graph(frames)

                for i in range(len(self.sw_sizes)):
                    self.queues_graph[i].put(st_graphs[i])

                # print("Push graph!")

                # wait for NN to prevent filling unnecessary many information into queue
                # self.queues_graph[0].join()

    def stop(self):
        self.is_stop = True


class NeuralNetworkThread(threading.Thread):
    """
    Thread for NeuralNetwork computing
    """

    def __init__(self, thread_id, network_number, sw_size, queue_graph, queue_mtm, device):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.network_number = network_number
        self.sw_size = sw_size
        self.queue_graph = queue_graph
        self.queue_mtm = queue_mtm

        self.action_nn = ActonDetectionNN(sw_size, device)

        self.is_stop = False

    def run(self):
        """
        Do cool stuff for action recognition here
        :return:
        """
        while not self.is_stop:
            # print("Run thread:{}".format(self.threadID))

            st_graph = self.queue_graph.get()
            # print("Thread NN (sw_size {}): st_graph = {}".format(self.sw_size, st_graph.shape))
            # print("start NN thread {} on graph".format(self.threadID))

            # ToDo: not used currently
            # set task done here to join STGraphThread to be one move ahead of NN that queue is never empty
            # self.queue_graph.task_done()

            action_primitive = self.action_nn.run_network(st_graph)

            # pass result of NN to MTM processing thread
            # ToDO: maybe only add if all processed (e.g. queue is empty)
            # action_primitive = [11]
            self.queue_mtm.put(action_primitive)
            # print("action primitive of NN thread {} : {}".format(self.threadID, action_primitive))

    def stop(self):
        self.is_stop = True


class MTMPreprocessingThread(threading.Thread):
    """
    Thread to handle incomming estimates from neural networks
    """

    def __init__(self, thread_id, queues_mtm, queue_mtm_preproc):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.queues_mtm = queues_mtm
        self.queue_mtm_preproc = queue_mtm_preproc

        self.is_stop = False

    def run(self):
        """
        preprocessing for MTMCompareThread
        :return:
        """

        action_primitives = [None] * len(self.queues_mtm)
        mtm_primitive = [0] * len(al.action_labels)
        queue_flag = False

        while not self.is_stop:
            # print("Run thread:{}".format(self.threadID))

            # print("Queue_flag:{}".format(queue_flag))

            # check for queue not zero
            for i in range(len(self.queues_mtm)):
                try:
                    action_primitive = self.queues_mtm[i].get(block=False)
                except queue.Empty:
                    continue
                queue_flag = True
                action_primitives[i] = action_primitive

            # check if queue is not empty and then use action primitives
            if not queue_flag:
                continue

            # get maximum out of each action class
            # ToDO: maybe use all max values of each tensor in mtm_primitive
            action_prob = [0] * len(al.action_labels)
            count = 0
            for i in range(len(self.queues_mtm)):
                if action_primitives[i] is None:
                    count += 1
                    continue
                # max_prob_nn = torch.max(action_primitives[i]).item()
                # max_prob_nn_idx = torch.argmax(action_primitives[i])
                # if max_prob_nn > action_prob[max_prob_nn_idx]:
                #     action_prob[max_prob_nn_idx] = max_prob_nn
                #     # action_est = torch.argmax(action_primitives[i])
                #     print(action_prob)

                for j in range(len(action_prob)):
                    action_prob[j] += action_primitives[i].tolist()[0][j]
            for i in range(len(action_prob)):
                action_prob[i] /= (len(self.queues_mtm) - count)
            mtm_primitive = action_prob

            # print("Thread MTM: action primitive = {}".format(al.get_key_by_value(action_est)))
            # print("Thread MTM: action primitive = {}".format(get_key_by_value(max(action_primitive_0))))

            # send new action estimates to MTM compare thread
            self.queue_mtm_preproc.put(mtm_primitive)
            queue_flag = False

    def stop(self):
        self.is_stop = True


class MTMCompareThread(threading.Thread):
    """
    Thread for comparing with MTM assembly plan and final estimation of assembly step
    """

    def __init__(self, thread_id, queue_mtm_preproc):
        threading.Thread.__init__(self)
        self.threadID = thread_id

        self.mtm_compare = MTMCompare()

        self.queue_mtm_preproc = queue_mtm_preproc

        self.is_stop = False

    def run(self):
        """
        Do cool stuff for MTM compare and final assembly step estimate here
        :return:
        """
        while not self.is_stop:
            # try to get preprocessed action estimate from queue and update search algorithm input
            try:
                mtm_primitive = self.queue_mtm_preproc.get(block=False)
                self.mtm_compare.update_action_est(mtm_primitive)
            except queue.Empty:
                pass

            # get final assembly step estimate
            step_est = self.mtm_compare.compare()

    def stop(self):
        self.is_stop = True


class ThreadController:

    def __init__(self):
        # create queues:
        # FIFO ques with predefined queue sizes
        # queue blocks insertion if full! If max size 0 -> queue is infinite
        queue_skeleton_size = 0
        queue_graph_size = 0
        queue_mtm_size = 0
        queue_mtm_preproc_size = 0
        # create queue for skeleton data (body tracking -> st graph generation)
        self.queue_skeleton = deque()

        # create queues from st graph generation to neural networks
        self.queue_graph_sw_0 = queue.Queue(queue_graph_size)
        self.queue_graph_sw_1 = queue.Queue(queue_graph_size)
        self.queue_graph_sw_2 = queue.Queue(queue_graph_size)
        self.queues_graph = [self.queue_graph_sw_0, self.queue_graph_sw_1, self.queue_graph_sw_2]

        # create queues for MTM primitives (neural networks -> mtm compare)
        self.queue_mtm_0 = queue.Queue(queue_mtm_size)
        self.queue_mtm_1 = queue.Queue(queue_mtm_size)
        self.queue_mtm_2 = queue.Queue(queue_mtm_size)
        self.queues_mtm = [self.queue_mtm_0, self.queue_mtm_1, self.queue_mtm_2]

        self.queue_mtm_preproc = queue.Queue(queue_mtm_preproc_size)

        # set sliding window sizes for NNs (must match with loaded weights !)
        self.sw_size_0 = 30
        self.sw_size_1 = 40
        self.sw_size_2 = 100
        self.sw_sizes = [self.sw_size_0, self.sw_size_1, self.sw_size_2]

        # set gpu device
        self.device = start_cuda()

        # create threads:
        # body tracking thread
        self.body_tracking_thread = BodyTrackingThread(0, self.queue_skeleton, self.sw_sizes)
        # graph thread
        self.graph_thread = STGraphThread(1, self.queue_skeleton, self.queues_graph, self.sw_sizes, self.device)

        # neural networks with different sliding window sizes
        self.nn_threads = []
        for i in range(0, len(self.sw_sizes)):
            self.nn_threads.append(
                NeuralNetworkThread(2 + i, i, self.sw_sizes[i], self.queues_graph[i], self.queues_mtm[i], self.device))

        # MTM compare threads:
        self.mtm_preprocessing_thread = MTMPreprocessingThread(2 + len(self.sw_sizes), self.queues_mtm,
                                                               self.queue_mtm_preproc)
        self.mtm_compare_thread = MTMCompareThread(3 + len(self.sw_sizes), self.queue_mtm_preproc)

        self.threads = [self.body_tracking_thread, self.graph_thread] + self.nn_threads + \
                       [self.mtm_preprocessing_thread, self.mtm_compare_thread]

    def start_threads(self):
        for thread in self.threads:
            thread.start()

    def stop_threads(self):
        # send stop command
        for thread in self.threads:
            thread.stop()

        # Wait for all threads to stop properly:
        for thread in self.threads:
            thread.join()

        print("All threads closed.")


if __name__ == '__main__':
    thread_ctrl = ThreadController()
    thread_ctrl.start_threads()
    input()
    thread_ctrl.stop_threads()
