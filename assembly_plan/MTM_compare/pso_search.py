import numpy as np
import random
import matplotlib.pyplot as plt
import math

from action_detection.SW_GCN.action_labels import action_labels


class PSOSearch:

    def __init__(self, swarm_size=100, neighborhood_size=100):
        """

        """
        # different versions:
        self.simple_pso = False
        self.inertia_w = False
        self.constriction_pso = True

        # velocity update params:

        # simple pso without inertia
        if self.simple_pso:
            self.weight = 0.792
            self.c_cog = 1.4944  # trust on own best (cognition)
            self.c_social = 1.4944  # trust on neighboring best (social)
            self.title = 'Simple PSO search'

        # inertia weight pso
        if self.inertia_w:
            self.weight = 1.1
            self.c_cog = 1.4944  # trust on own best (cognition)
            self.c_social = 1.4944  # trust on neighboring best (social)
            self.title = 'Inertia weight PSO search'

        # constriction factor pso
        if self.constriction_pso:
            self.weight = 0.3  # particle weight for inertia
            self.c_cog = 2.05  # values from paper for good convergence and guaranteed stability
            self.c_social = 2.05
            self.c_phi = self.c_cog + self.c_social  # for constriction pso (>4 to guarantee stability)
            self.k = 2 / (abs(2 - self.c_phi - math.sqrt(self.c_phi * self.c_phi - 4 * self.c_phi)))
            self.title = 'Constriction factor PSO search'

        self.vel_init = 1  # maximum magnitude of initial velocity per dimension


        # search params
        self.iteration = 0

        # swarm
        self.swarm_size = swarm_size
        # using a ring topology for neighborhood (social neighbors according to position in self.population)
        self.neighborhood_size = neighborhood_size

        # swarm:
        # list of individuals: each individual [current, velocity, personal best, global best]
        # current: (pos_x,pos_y, fitness)
        # velocity: (vel_x,vel_y)
        # best as position tuple
        self.swarm = []
        self.initialize_swarm()

        # store global best
        self.global_best = self.swarm[0][0]
        self.iteration_best_found = self.iteration
        self.best_of_iteration = []
        self.best_of_swarm()

        # store average fitness of iteration
        self.fitness_avg_iteration = []
        self.calc_avg_fit()

    def initialize_swarm(self, save_prev_best=False):
        """
        initialize swarm with particles
        :return:
        """
        self.swarm = []
        for i in range(self.swarm_size):
            if save_prev_best and i == 0 and len(self.best_of_iteration) != 0:
                pos_x = self.best_of_iteration[-1][0][0]
            else:
                # initialize position randomly in range
                pos_x = random.randint(self.x_min, self.x_max)
            fit = self.fitness(pos_x)
            pos = (pos_x, fit)

            # initialize velocity (zero or small values to prevent big jumps from initial position)
            vel = (random.uniform(-self.vel_init, self.vel_init))

            # personal/global best = initial solution
            p_best = pos
            g_best = pos

            particle = [pos, vel, p_best, g_best]

            self.swarm.append(particle)


    def update_velocity(self, p, particle_index):
        """
        update rule for particle velocity
        :return:
        """
        # random numbers for cognitive and social
        r_cog = random.random()
        r_social = random.random()

        # calculate weight for inertia weight version
        if self.inertia_w:
            self.calc_w(p)

        # calculate velocity of particle
        vel = self.weight * p[1] + self.c_cog * r_cog * (p[2][0] - p[0][0]) + self.c_social * r_social * (
                p[3][0] - p[0][0])

        # constriction factor pso
        if self.constriction_pso:
            vel = self.k * vel

        return vel

    def update_pos(self, particle_index):
        """
        update position of particle according to its velocity
        :return:
        """
        p = self.swarm[particle_index]  # particle

        vel = self.update_velocity(p, particle_index)
        pos_x = p[0][0] + vel
        # keep position in bounds
        pos_x = self.pos_bounds(pos_x)

        fit = self.fitness(pos_x)
        pos = (pos_x, fit)

        self.swarm[particle_index][0] = pos

        # check if personal best
        if fit < p[2][1]:
            self.swarm[particle_index][2] = pos

    def pos_bounds(self, pos):
        """
        - ensures new position in bounds
        - rounds position to integer value
        :param pos:
        :return:
        """
        pos_x = round(pos)

        if pos_x > self.x_max:
            pos_x = self.x_max
        elif pos_x < self.x_min:
            pos_x = self.x_min

        return pos_x

    def fitness(self, pos):
        """
        calculate fitness of particles solution (position)

        objective function:

        :param pos: position pos = pos_x
        :return:
        """

        return pos

    def best_in_neighborhood(self, particle_index):
        """
        finds best solution in neighborhood
        :return:
        """
        # cost of global best stored in particle
        fit_best = self.swarm[particle_index][3][1]

        # also check particle itself to update g_best if p_best is higher
        for i in range(self.neighborhood_size):

            # force wraparound when reaching end of swarm list
            n = particle_index + i
            if n >= len(self.swarm):
                n = n - len(self.swarm)

            # update particle best if found better in personal bests of neighborhood
            if self.swarm[n][2][1] < fit_best:
                # memorize fitness
                fit_best = self.swarm[n][2][1]

                # store p_best from neighbor to g_best of particle
                self.swarm[particle_index][3] = self.swarm[n][2]

    def best_of_swarm(self, reset_global_best=False):
        """
        find best particle of current iteration in swarm (current best not p_best)
        :return: best particle of current iteration
        """
        best = self.swarm[0][0]
        best_idx = 0
        for i in range(len(self.swarm)):
            if best[1] > self.swarm[i][0][1]:
                best = self.swarm[i][0]
                best_idx = i

        # check if best of iteration is better than best ever found
        if best[1] < self.global_best[1] or reset_global_best:
            self.global_best = best
            self.iteration_best_found = self.iteration

        # store in list of best per iteration
        self.best_of_iteration.append([best, best_idx])

    def calc_avg_fit(self):
        """
        calulate the average fitness of an iteration
        :return:
        """
        fit = 0
        for i in range(len(self.swarm)):
            fit += self.swarm[i][0][1]

        self.fitness_avg_iteration.append(fit / len(self.swarm))

    def calc_w(self, p):
        """
        calculate weight
        :return:
        """
        # weight = (1.1 - (fit(g_best)/fit(p_best)))
        self.weight = 1.1 - (self.fitness(p[3][0]) / self.fitness(p[2][0]))

    def count_suc_fail(self, success=True, reset=False):
        """
        count success and failure of best particle for guaranteed convergence PSO
        :return:
        """
        if reset:
            self.sc_num = 0
            self.f_num = 0
            return

        if success:
            self.sc_num += 1
            self.f_num = 0
        else:
            self.sc_num = 0
            self.f_num += 1

    def print_swarm(self):
        for i in range(len(self.swarm)):
            print(self.swarm[i])
        print('\n')

    def plot_search(self):

        plt.subplot(122)
        x = [el[0][0] for el in self.swarm]
        y = [el[0][1] for el in self.swarm]
        color = ['m']
        plt.scatter(x, y, s=100, marker='.', c=color)

        opt_x = [-0.089840, 0.089840]
        opt_y = [0.712659, -0.712659]
        plt.scatter(opt_x, opt_y, s=100, marker='+', c='r')

        plt.grid(True)
        plt.title(self.title)
        # plt.legend(['otimum'])
        plt.ylabel('y')
        plt.xlabel('x')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        plt.pause(0.05)

    def stop_search(self):
        """
        doing some prints/plots and then exit
        :return:
        """
        # prints
        print("------------------------------------------------------------")
        print("Stop search!")
        print("Iterations: {}".format(self.iteration))
        print("Best Solution with cost {} found at iteration {} :".format(self.global_best[2],
                                                                          self.iteration_best_found))
        print(self.global_best)

        # plot
        # plt.figure(1)

        # annotate scatter plot
        txt = "Best Solution:\n x={}\n y={}\n fitness={}".format(self.global_best[0], self.global_best[1],
                                                                 self.global_best[2])
        plt.annotate(txt, (-4, -4))
        # average fitness
        plt.subplot(321)
        plt.plot(self.fitness_avg_iteration, 'r-')
        plt.ylabel('fitness value')
        plt.xlabel('Iterations')
        plt.title('Fitness over iterations')
        plt.legend(['fitness_average'])
        plt.grid(True)

        # best fitness
        plt.subplot(325)
        plt.plot([el[0][2] for el in self.best_of_iteration], 'b-')
        plt.ylabel('fitness value')
        plt.xlabel('Iterations')
        plt.title('Fitness of best particle per iteration')
        plt.legend(['fitness_best'])
        plt.grid(True)

        plt.show()

    def search(self):
        """
        main loop performing PSO search
        :return:
        """

        self.iteration += 1

        # iterate over population
        for i in range(len(self.swarm)):
            # asynchronous update
            if not self.simple_pso:
                self.best_in_neighborhood(i)

            self.update_pos(i)

        if self.simple_pso:
            for i in range(len(self.swarm)):
                # synchronous update - simple PSO
                self.best_in_neighborhood(i)

        # get best out of iteration
        self.best_of_swarm()

        self.calc_avg_fit()



        # return best found solution
        return self.global_best[0]


if __name__ == '__main__':
    # pop_size = 10
    # pso = PSOSearch(pop_size)
    # pso.print_pop()
    # pso.best_in_neighborhood(0)
    # pso.print_pop()
    #
    # pso.update_pos(0)
    #
    # pso.print_pop()

    pso = PSOSearch()
    pso.search()
    # print(pso.fitness((-0.089840, -0.712659)))
