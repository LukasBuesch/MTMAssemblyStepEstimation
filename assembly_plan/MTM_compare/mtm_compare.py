"""
compare detected MTM primitives with assembly plan

- also include MTM plan weights here
- generate live assembly step estimate

"""
import json
import time

# import torch
from collections import Counter

import action_detection.SW_GCN.action_labels as al
from assembly_plan.MTM_compare.IO_for_txt_files import read_from_txt
from assembly_plan.MTM_compare.pso_search import PSOSearch


class MTMCompare(PSOSearch):

    def __init__(self):
        # import settings
        # with open("MTM_assembly_plan_config.json") as json_file:
        with open("assembly_plan\\MTM_compare\\MTM_assembly_plan_config.json") as json_file:
            settings = json.load(json_file)

        # load assembly plan
        self.assembly_plan = read_from_txt(settings["assembly_plan"], settings["assembly_plans"])
        # self.assembly_plan = read_from_txt(settings["assembly_plan_test"], settings["assembly_plans_test"])

        len_assembly_plan = int((len(self.assembly_plan) / 2))

        self.assembly_plan_primitives = [None] * len_assembly_plan
        self.assembly_plan_times = [None] * len_assembly_plan

        count_1 = 0
        count_2 = 0
        for i in range(len(self.assembly_plan)):
            if i % 2 == 0:
                self.assembly_plan_primitives[count_1] = al.action_labels[self.assembly_plan[i]]
                count_1 += 1
            else:
                self.assembly_plan_times[count_2] = float(self.assembly_plan[i])
                count_2 += 1

        # list with probabilities of each action class
        self.action_est_nn = [0] * len(al.action_labels)
        self.action_est = 0
        self.action_est_hist = []
        self.action_est_hist.append(self.action_est)
        self.action_est_prob = 0
        self.action_est_nn_hist = []
        self.action_est_nn_hist.append(self.action_est_nn)

        # objective function constraints for solution
        self.x_max = len(self.assembly_plan_primitives) - 1
        self.x_min = 0

        # fitness tuning values
        self.alpha_mtm_hist_dist = 1.0
        self.alpha_time_dist = 10.0
        self.action_est_prob_threshold = 0.1
        self.alpha_action_primitive = 0.5

        # time tracking
        self.start_time = time.time()

        # run __init__ of superclass
        PSOSearch.__init__(self, swarm_size=50, neighborhood_size=2)

    def compare(self):
        """
        run PSO algorithm and update self.action_est from MTM thread
        :return: assembly step estimate as step number
        """
        # ToDo: hotfix to update swarm for following time optima -> replace by only replacing subset of worst particle
        self.initialize_swarm(save_prev_best=True)
        self.best_of_swarm(reset_global_best=True)
        step_est = self.search()
        # if self.iteration == 1:
        #     self.print_solution()
        self.print_solution()
        return step_est

    def update_action_est(self, action_est_nn):
        """
        - update action estimate
        - store last estimate to action estimation history
        :param action_est_nn:
        :return:
        """
        self.action_est_nn_hist.append(action_est_nn)
        self.action_est_nn = action_est_nn
        self.action_est_prob = max(action_est_nn)
        action_est = action_est_nn.index(self.action_est_prob)
        if action_est != self.action_est:
            self.action_est = action_est
            self.action_est_hist.append(action_est)

        # reset search to handle new information

        self.initialize_swarm(save_prev_best=True)
        self.best_of_swarm(reset_global_best=True)
        self.iteration = 0
        self.iteration_best_found = self.iteration
        self.best_of_iteration = []

    def fitness(self, pos_x):
        """
        calculate fitness of particles solution (position)
        -> low fitness is goal

        objective function:

        :param pos_x: position of particle in search space
        :return:
        """
        fitness = 0

        # check how long history of MTM primitive estimates and give low value for estimate near time steps
        mtm_hist_est = len(self.action_est_hist) - 1

        mtm_hist_dist = abs(mtm_hist_est - pos_x)

        fitness += self.alpha_mtm_hist_dist * mtm_hist_dist

        # check if current solutions fits to runtime based on execution times
        time_dist = abs(self.estimate_time_from_plan() - pos_x)
        fitness += self.alpha_time_dist * time_dist

        # check if current action primitive of assembly plan matches current action primitive from NN
        if not self.assembly_plan_primitives[pos_x] == self.action_est:
            # print("action est {}".format(self.action_est))
            # prevent to large influence of very small probabilities
            if self.action_est_prob < self.action_est_prob_threshold:
                action_est_prob = self.action_est_prob_threshold
            else:
                action_est_prob = self.action_est_prob
            fitness += self.alpha_action_primitive / action_est_prob  # action_est_prob = 0...1

        return fitness

    def print_solution(self):
        print("\ntime: {}".format(time.time() - self.start_time))
        print("estimated actions: {}".format(self.action_est_nn_hist))
        print("len(estimated actions): {}".format(len(self.action_est_nn_hist)))
        print("estimated actions (trimmed): {}".format(self.action_est_hist))
        print("len(estimated actions) (trimmed): {}".format(len(self.action_est_hist)))
        print("PSO iteration: {}\n".format(self.iteration))

        particle_action_counter = Counter([el[0][0] for el in self.swarm])
        # print("global_best {}".format(self.global_best))

        for i in range(len(self.assembly_plan_primitives)):
            if i == self.global_best[0]:
                print("X ", end='')
            else:
                print("  ", end='')
            print("Assembly step {}: {} (particles/fitness:{}/{})".format(i,
                                                                          al.get_key_by_value(
                                                                              self.assembly_plan_primitives[i]),
                                                                          particle_action_counter[i], self.fitness(i)))
        print("---------------------------------------------------------")

    def estimate_time_from_plan(self):
        # current time
        current_time = time.time() - self.start_time
        time_acc = 0
        for i in range(len(self.assembly_plan_times)):
            time_acc += self.assembly_plan_times[i]
            if time_acc > current_time:
                return i
        return len(self.assembly_plan_times)


if __name__ == '__main__':
    m = MTMCompare()
    m.compare()
