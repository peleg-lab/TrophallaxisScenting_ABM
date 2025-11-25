import mesa
import numpy as np
import pandas as pd
import math
import random as rand

import h5py

from .bee import Bee
from .environment import Environment

import datacollectors

import globals

class TrophallaxisABM(mesa.Model):

    def __init__(self, session_id=0, batch_id=0, run_id=0,
                ## Diffusion Parameters:
                A_=1.0, Dc=0.06, wb_=20.0, del_t=0.005, dec_rate=5.0,
                ## Scenting Parameters:
                scent_thresh=0.15, scent_prob=0.33, scent_freq=80, scent_move=True,
                ## Fanning Parameters:
                fan_threshold=0.15, trans_prob_=0.5,
                ## Bee Parameters:
                N=110, fraction_of_fed_bees=10, theta=180,
                ## Arena Parameters:
                solid_bounds=True, data_out=False,
                width=32, height=32):
        ## Super:
        super().__init__()

        ## Initialize globals:
        globals.n3_counter = 0
        globals.n2_counter = 0
        globals.n1_counter = 0
        globals.max_transfer_t = 50
        globals.food_transfer_rate = 0.5 / globals.max_transfer_t
        globals.newvar = 0.1
        globals.var = 0.0
        globals.epsilon = 10**-8    #10**-eps_boost

        ## init vars:
        self.N = N
        self.fraction_of_fed_bees = fraction_of_fed_bees
        self.theta = np.deg2rad(theta)
        self.width = width
        self.height = height

        ## End check vars:
        self.end_check = 0.0001
        self.troph_thresh = self.end_check * 10      #self.end_check * a_boost
        self.att_boost = 10     #a_boost
        self.end_boost = 8      #end_boost
        self.all_fed = False

        ## Scenting:
        self.environment = Environment(0, 32, 0.1, dec_rate, Dc, del_t)
        self.environment.init_concentration_map()
        self.old_walk = False       # set TRUE to not do gradient ascent walk
        self.fed_scent_threshold = scent_thresh         # amount of food needed to scent
        self.fed_scent_prob = scent_prob                # probability of scenting if food > scent_thresh
        self.fed_scent_freq = scent_freq                # emission frequency for fed bees
        self.fed_scent_move = scent_move                # (boolean) if fed bees can move during wait time

        ## Scheduler:
        self.schedule = mesa.time.RandomActivation(self)

        ## Space:
        self.space = mesa.space.ContinuousSpace(self.width, self.height, torus=(not solid_bounds))
        self.is_toroidal = not solid_bounds
        self.bee_positions = {}

        ## For analysis:
        self.output_data = data_out
        self.t_i = 0
        self.environment_history = []
        self.no_pheromones = False           # enable to test only random walking agents (no scenting)

        ## Parameter sweep:
        self.data = {}
        self.food_data = {}
        self.scent_data = {}
        self.session_id = session_id
        self.batch_id = batch_id
        self.run_id = run_id #globals.run_counter
        self.save_parameter_log(dec_rate, Dc, del_t, wb_, A_, fan_threshold, trans_prob_, scent_thresh, scent_prob, scent_freq)

        ###
        self.cull_counts = []
        ###

        ## add bee agents:
        self.initiate_bees(A_,
                           wb_,
                           fan_threshold,
                           trans_prob_)

        ## DataCollector:
        self.data_collector = mesa.DataCollector(
            model_reporters={"Fed .50": datacollectors.num_fed_50,
                             "Fed .25": datacollectors.num_fed_25,
                             "Fed .12": datacollectors.num_fed_12,
                             "Fed .05": datacollectors.num_fed_05,
                             "Fed .01": datacollectors.num_fed_01,
                             "Fed 0.0": datacollectors.num_fed_00,
                            }
        )

        ## initialize CSV output file:
        # if self.output_data:
        self.initialize_data()
        self.write_cMap()        # save initial concentration map
    # __init__()

    def step(self):
        ## Invrement counter and reset tracker:
        self.t_i += 1
        self.all_fed = True

        ## Step 1: Build pheromone sources list for current timestep
        ## Update concentrations from queens and workers:
        bee_agents = self.agents.select(lambda agent: agent.type == 0)
        if not self.no_pheromones:
            for a in bee_agents:
                self.environment.update_pheromone_sources(a, self.bee_positions, self.t_i)

        # queens = self.agents.select(lambda agent: agent.is_queen)
        # for q in queens:
        #     self.environment.update_pheromone_sources(q, self.t_i)
        # workers = self.agents.select(lambda agent: agent.is_queen == False)
        # for w in workers:
        #     self.environment.update_pheromone_sources(w, self.t_i)

        ## Cull pheromones:
        cnt = self.environment.cull_pheromone_sources(self.t_i)
        self.cull_counts.append(cnt) # [remove]
        if self.t_i%50==0:
            print(self.cull_counts)
            self.cull_counts = []

        ## Init concentration map:
        self.environment.init_concentration_map()

        ## Step 2: Build concentration map and get gradients
        for pheromone_src in self.environment.pheromone_sources:
            ## update concentration map with x, y, A, dt, etc.
            pheromone_src_C = self.environment.update_concentration_map(self.t_i, pheromone_src)

            ## Iterate through list of active bees and calculate gradient from current source
            for bee in bee_agents:
                bee.sense_environment(self.environment, pheromone_src, pheromone_src_C)

        ## Step 3: Bee Movements
        ## Do agent steps:
        # #             -1 0 1 2 3 4
        state_counter = [0,0,0,0,0,0]
        fed_counter = 0
        fed_update_counter = [0,0]
        for a in bee_agents:
            if a.food > self.fed_scent_threshold:
                a.fed_update()
                fed_update_counter[0] += 1
            else:
                a.update()
                fed_update_counter[1] += 1
        for a in bee_agents:
            # if a.food == 0:     ### [ REMOVE THIS -> fixed queen run ]
            state_counter[a.state] += 1
            a.step()
            if a.food > 0:
                fed_counter += 1

        # #######[ TEMPORAL PRINT ]#######
        # if self.t_i%5 == 0:
        #     # print(str(self.t_i) + " - " + str(fed_counter))
        #     # print(" - r_" + str(self.run_id) + ": " + str(self.t_i) + " - " + str(fed_counter) + " | " + str(fed_update_counter[0]) + " , " + str(fed_update_counter[1]))
        #     print("#",end="")
        #     # print(len(self.environment.pheromone_sources))
        # #######[ TEMPORAL PRINT ]#######

        ## Collect data:
        self.data_collector.collect(self)
        # if self.output_data:
        self.write_cMap()            # save concentration map
        self.write_data(self.t_i)

        ## Check stopping condition:
        globals.deltavar = np.abs(globals.newvar - globals.var)
        if self.all_fed:
            self.check_stop_condition()
        if self.t_i >= 300:
            self.running = False
            print("- overtime -")
            self.save_data()
            self.save_food()
            self.save_scent()
            self.save_cMap()
        if not self.running:
            return
    # step()

    def check_stop_condition(self):
        if globals.deltavar < globals.epsilon and globals.newvar < (self.end_check * self.end_boost):
            # if self.output_data:
            self.running = False
            self.save_data()
            self.save_food()
            self.save_scent()
            self.save_cMap()
    # check_stop_condition()

    def initiate_bees(self, A_, wb_, fan_threshold, trans_prob_):
        ## find starting points:
        points = [[x,y] for x in range(1,self.width) for y in range(1,self.height)]
        rand.shuffle(points)

        ## initiate bee agents:
        for i in range(self.N):
            loc = points[i]
            b = Bee(i, self, float(loc[0]), float(loc[1]), A_, wb_, fan_threshold, trans_prob_)
            self.bee_positions[f'bee_{b.unique_id}'] = [b.x, b.y]
            if i < (self.N * self.fraction_of_fed_bees/100.0):
                b.make_fed()
            self.schedule.add(b)
            self.space.place_agent(b, loc)
    # initiate_bees()

    def save_parameter_log(self, decay_rate, Dc, del_t, wb, A, fan_thresh, trans_prob, fed_thresh, fed_prob, fed_freq):
        path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/a_paramlog.txt"
        f = open(path, "w")
        f.write("Session: "+str(self.session_id)+" Batch: "+str(self.batch_id)+
                "\nN: "+str(self.N)+"\nfrac_fed: "+str(self.fraction_of_fed_bees)+"\ntheta: "+str(self.theta)+
                "\ndecay_rate: "+str(decay_rate)+"\nD: "+str(Dc)+"\ndelta_t: "+str(del_t)+"\nwb: "+str(wb)+"\ninitial_concentration: "+str(A)+"\nscenting_threshold: "+str(fan_thresh)+
                "\nscenting_prob: "+str(trans_prob)+"\nfed_scent_threshold: "+str(fed_thresh)+"\nfed_scent_prob: "+str(fed_prob)+
                "\nfed_scent_frequency: "+str(fed_freq))
        f.close()
    # save_parameter_log()

    def initialize_data(self):
        xs = []
        ys = []
        food = []
        scent = []
        bee_agents = self.agents.select(lambda agent: agent.type == 0)
        for a in bee_agents:
            xs.append(a.pos[0])
            ys.append(a.pos[1])
            food.append(a.food)
            scent.append(1 if a.state==2 else 0)
        self.data = {
            "x_0": xs,
            "y_0": ys
        }
        self.food_data = {
            0: food
        }
        self.scent_data = {
            0: scent
        }
    # initialize_data()

    def write_data(self, step):
        xs = []
        ys = []
        food = []
        scent = []
        bee_agents = self.agents.select(lambda agent: agent.type == 0)
        for a in bee_agents:
            coords = self.bee_positions[f'bee_{a.unique_id}']
            xs.append(coords[0])
            ys.append(coords[1])
            food.append(a.food)
            scent.append(1 if (a.state==1 or a.state==4) else 0)
        self.data["x_" + str(step)] = xs
        self.data["y_" + str(step)] = ys
        self.food_data[step] = food
        self.scent_data[step] = scent
    # write_data()

    def save_data(self):
        df = pd.DataFrame(self.data)
        path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/run_" + str(self.run_id) + ".csv"
        # path = "data_out/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/run_" + str(self.run_id) + ".csv"
        df.to_csv(path)
    # save_data()

    def save_food(self):
        df = pd.DataFrame(self.food_data)
        path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/food_" + str(self.run_id) + ".csv"
        # path = "data_out/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/food_" + str(self.run_id) + ".csv"
        df.to_csv(path)
    # save_food()

    def save_scent(self):
        df = pd.DataFrame(self.scent_data)
        path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/scent_" + str(self.run_id) + ".csv"
        # path = "data_out/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/scent_" + str(self.run_id) + ".csv"
        df.to_csv(path)
    # save_scent()

    def write_cMap(self):
        self.environment_history.append(self.environment.concentration_map)
    # write_cMap()

    def save_cMap(self):
        with h5py.File(r"/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/cMap_" + str(self.run_id) + ".h5", 'w') as outfile:
        # with h5py.File(r"data_out/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/cMap_" + str(self.run_id) + ".h5", 'w') as outfile:
            outfile.create_dataset("concentration", data=self.environment_history)
    # save_cMap()

# Class TrophallaxisABM()
