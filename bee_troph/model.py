import mesa
import numpy as np
import pandas as pd
import math
import random as rand
from pathlib import Path
import h5py

from .bee import Bee
from .environment import Environment

import datacollectors

import globals

IS_LOCAL = True

class TrophallaxisABM(mesa.Model):

    def __init__(self, session_id=0, batch_id=0, run_id=0,
                ## Diffusion Parameters:
                Dc=0.06, wb_=20.0, del_t=0.005, dec_rate=5.0, 
                cull_thresh=0.0001,
                ## Scenting Parameters:
                fed_scent_threshold_=0.15, fed_scent_prob_=0.33, fed_scent_freq_=80, fed_skip_wait_=True,
                ## Fanning Parameters:
                unfed_threshold_=0.15, unfed_trans_prob_=0.5,
                ## Bee Parameters:
                N=110, number_of_fed_bees=10, theta=180,
                ## Special Modes/testing options:
                use_queens_=False, time_limit_=400, noisey_directed_walk=False,
                ## Arena Parameters:
                solid_bounds=True, output_data_=True,
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
        globals.epsilon = 10**-8

        ## init vars:
        self.N = N
        self.number_of_fed_bees = number_of_fed_bees
        self.theta = np.deg2rad(theta)
        self.width = width
        self.height = height

        ## End check vars:
        self.end_check = 0.0001    # compared to newvar
        self.end_boost = 8         # if (newvar < end_check * end_boost)
        # self.troph_thresh = self.end_check * 10                                                   # [REMOVE]
        self.all_fed = False
        self.time_limit = time_limit_

        ## Scenting:
        self.environment = Environment(0, 32, 0.1, dec_rate, Dc, del_t, cull_thresh)
        self.environment.init_concentration_map()
        # self.old_walk = False       # set TRUE to not do gradient ascent walk                     [REMOVE]
        self.fed_scent_threshold = fed_scent_threshold_         # amount of food needed to scent
        self.fed_scent_prob = fed_scent_prob_                # probability of scenting if food > scent_thresh
        self.fed_scent_freq = fed_scent_freq_                # emission frequency for fed bees
        self.fed_skip_wait = fed_skip_wait_                # (boolean) if fed bees can move during wait time

        ## [ REMOVE the following later: ]
        self.num_scenting = 0       # number of bees currently scenting (maybe remove later?)
        self.max_scenting_bees = int(self.N * 0.10)   # max number of bees that can scent at once
        ## [ REMOVE the above later: ]

        ## Scheduler:
        self.schedule = mesa.time.RandomActivation(self)

        ## Space:
        self.space = mesa.space.ContinuousSpace(self.width, self.height, torus=(not solid_bounds))
        self.is_toroidal = not solid_bounds
        self.bee_positions = {}

        ## For analysis:
        self.output_data = output_data_
        self.t_i = 0
        self.environment_history = []

        ## Special Modes:
        self.use_queens = use_queens_           # Queen Finding Mode:
        self.disable_trophallaxis = False       #  - Disable food exchange
        self.fed_cant_move = False              #  - Fed bees cannot move
        ###### TESTING OPTIONS: ######
        self.single_bee_test = False  #
        ##############################
        # self.no_pheromones = False              #  - No pheromone emission / sensing           # enable to test only random walking agents (no scenting)
        self.noise_in_directed_walk = noisey_directed_walk   # add noise to directed walk
        self.check_run_mode()

        ## Parameter sweep:
        self.data = {}
        self.grad_data = {}
        self.food_data = {}
        self.scent_data = {}
        self.session_id = session_id
        self.batch_id = batch_id
        self.run_id = run_id #globals.run_counter
        A_ = 0.0575
        self.save_parameter_log(dec_rate, Dc, del_t, wb_, A_, unfed_threshold_, unfed_trans_prob_, fed_scent_threshold_, fed_scent_prob_, fed_scent_freq_, cull_thresh)

        #### TO BE REMOVED LATER ####
        self.grads_out = []
        self.Cs_out = []
        self.total_grads_out = []
        #### ## ## ####### ##### ####

        # ###
        # self.cull_counts = []       ## [REMOVE]
        # ###

        ## add bee agents:
        if self.single_bee_test:
            self.initiate_single_bee(wb_,
                                     unfed_threshold_,
                                     unfed_trans_prob_)
        else:
            self.initiate_bees(wb_,
                           unfed_threshold_,
                           unfed_trans_prob_)

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
        if self.output_data:
            self.initialize_data()
            self.write_cMap()        # save initial concentration map
            # self.write_comparison_data()   # save initial comparison data [ REMOVE LATER ]
    # __init__()

    def step(self):
        ## Increment counter and reset tracker:
        self.t_i += 1
        self.all_fed = True

        ## Step 1: Build pheromone sources list for current timestep
        ## Update concentrations from queens and workers:
        bee_agents = self.agents.select(lambda agent: agent.type == 0)
        # if not self.no_pheromones:
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
        # ## output cull counts for debugging:
        # self.cull_counts.append(cnt)            # [REMOVE]
        # if self.t_i%20==0:
        #     if len(self.cull_counts) > 0:
        #         print(self.cull_counts)
        #     self.cull_counts = []

        ## Init concentration map:
        self.environment.init_concentration_map()

        ## Step 2: Build concentration map and get gradients
        for pheromone_src in self.environment.pheromone_sources:
            ## update concentration map with x, y, A, dt, etc.
            pheromone_src_C = self.environment.update_concentration_map(self.t_i, pheromone_src)

            ## Iterate through list of active bees and calculate gradient from current source
            # for bee in bee_agents:
            #     bee.sense_environment(self.environment, pheromone_src, pheromone_src_C)

        ## Step 3: Bee Movements
        ## Do agent steps:
        # #             -1 0 1 2 3 4
        state_counter = [0,0,0,0,0,0]
        fed_counter = 0
        fed_update_counter = [0,0]
        for a in bee_agents:
            if a.food > self.fed_scent_threshold:
                if self.fed_cant_move:   ## queen mode
                    a.queen_update()
                else:
                    a.fed_update()
                fed_update_counter[0] += 1
            else:
                a.update()
                fed_update_counter[1] += 1
        for a in bee_agents:
            state_counter[a.state] += 1
            if self.fed_cant_move and a.food > self.fed_scent_threshold:    ## queen mode
                a.step_queen()
            else:
                a.step()
            if a.food > 0:
                fed_counter += 1
        
        self.num_scenting = state_counter[1]    # count number of unfed bees that are scenting

        # # #######[ TEMPORAL PRINT ]#######
        # if self.t_i%50 == 0:
        #     print(f'Current time: {self.t_i}')
        # #     # print(" - r_" + str(self.run_id) + ": " + str(self.t_i) + " - " + str(fed_counter) + " | " + str(fed_update_counter[0]) + " , " + str(fed_update_counter[1]))
        # #     print("#",end="")
        # #     # print(len(self.environment.pheromone_sources))
        # # #######[ TEMPORAL PRINT ]#######

        ## Collect data:
        self.data_collector.collect(self)
        if self.output_data:
            self.write_cMap()            # save concentration map
            self.write_data(self.t_i)
            # self.write_comparison_data()   # save comparison data [ REMOVE LATER ]

        ## Check stopping condition:
        globals.deltavar = np.abs(globals.newvar - globals.var)
        if self.all_fed and not self.disable_trophallaxis:
            self.check_stop_condition()
        if self.t_i >= self.time_limit:
            self.running = False
            print("- overtime -")
            self.save_data()
            self.save_grad_data()
            self.save_food()
            self.save_scent()
            self.save_cMap()
            # self.save_comparison_data()   # save comparison data [ REMOVE LATER ]
        if not self.running:
            return
    # step()

    def check_stop_condition(self):
        if globals.deltavar < globals.epsilon and globals.newvar < (self.end_check * self.end_boost):
            self.running = False
            if self.output_data:
                self.save_data()
                self.save_food()
                self.save_scent()
                self.save_cMap()
    # check_stop_condition()

    def initiate_bees(self, wb_, fan_threshold, trans_prob_):
        ## find starting points:
        points = [[x,y] for x in range(1,self.width) for y in range(1,self.height)]
        rand.shuffle(points)

        ## initiate bee agents:
        for i in range(self.N):
            loc = points[i]
            b = Bee(i, self, float(loc[0]), float(loc[1]), wb_, fan_threshold, trans_prob_)
            self.bee_positions[f'bee_{b.unique_id}'] = [b.x, b.y]
            # if i < (self.N * self.fraction_of_fed_bees/100.0):
            if i < self.number_of_fed_bees:
                b.make_fed()
            self.schedule.add(b)
            self.space.place_agent(b, loc)
    # initiate_bees()

    def initiate_single_bee(self, wb_, fan_threshold, trans_prob_):
        ## initiate ONE bee agent:
        b = Bee(0, self, float(16), float(16), wb_, fan_threshold, trans_prob_)
        b.heading = 0.0
        b.state = 1
        b.dont_move = True
        self.bee_positions[f'bee_{b.unique_id}'] = [b.x, b.y]
        self.schedule.add(b)
        self.space.place_agent(b, (16,16))
    # initiate_single_bee()

    def check_run_mode(self):
        if self.use_queens:
            self.fed_cant_move = True
            self.disable_trophallaxis = True
            self.fed_scent_prob = 1.0
            print("=== QUEEN MODE ENABLED ===")
    # check_run_mode()

    def save_parameter_log(self, decay_rate, Dc, del_t, wb, A, fan_thresh, trans_prob, fed_thresh, fed_prob, fed_freq, culling_threshold):
        # path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/a_paramlog.txt"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{self.session_id}/batch_{self.batch_id}/a_paramlog.txt"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{self.session_id}/batch_{self.batch_id}/a_paramlog.txt"

        fpath = Path(path)
        if fpath.is_file():
            print("NOTE: Parameter log already exists")
        else:
            f = open(path, "w")
            f.write("Session: "+str(self.session_id)+" Batch: "+str(self.batch_id)+
                    # "\nN: "+str(self.N)+"\nfrac_fed: "+str(self.fraction_of_fed_bees)+"\ntheta: "+str(self.theta)+
                    "\nN: "+str(self.N)+"\nnum_fed: "+str(self.number_of_fed_bees)+"\ntheta: "+str(self.theta)+
                    "\ndecay_rate: "+str(decay_rate)+"\nD: "+str(Dc)+"\ndelta_t: "+str(del_t)+"\nwb: "+str(wb)+"\ninitial_concentration: "+str(A)+"\nscenting_threshold: "+str(fan_thresh)+
                    "\nculling_threshold: "+str(culling_threshold)+
                    "\nscenting_prob: "+str(trans_prob)+"\nfed_scent_threshold: "+str(fed_thresh)+"\nfed_scent_prob: "+str(fed_prob)+
                    "\nfed_scent_frequency: "+str(fed_freq)+"\ntime_limit: "+str(self.time_limit))
            f.close()
    # save_parameter_log()

    def initialize_data(self):
        xs = []
        ys = []
        dx = []
        dy = []
        food = []
        scent = []
        bee_agents = self.agents.select(lambda agent: agent.type == 0)
        for a in bee_agents:
            xs.append(a.pos[0])
            ys.append(a.pos[1])
            dx.append(a.grad_x)
            dy.append(a.grad_y)
            food.append(a.food)
            scent.append(1 if a.state==1 else 0)    ## [ERROR with state]
        self.data = {
            "x_0": xs,
            "y_0": ys
        }
        self.grad_data = {
            "dx_0": dx,
            "dy_0": dy
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
        dx = []
        dy = []
        food = []
        scent = []
        bee_agents = self.agents.select(lambda agent: agent.type == 0)
        for a in bee_agents:
            coords = self.bee_positions[f'bee_{a.unique_id}']
            xs.append(coords[0])
            ys.append(coords[1])
            dx.append(a.grad_x)
            dy.append(a.grad_y)
            food.append(a.food)
            scent.append(1 if (a.state==1 or a.state==4) else 0)
        self.data["x_" + str(step)] = xs
        self.data["y_" + str(step)] = ys
        self.grad_data["dx_" + str(step)] = dx
        self.grad_data["dy_" + str(step)] = dy
        self.food_data[step] = food
        self.scent_data[step] = scent
    # write_data()

    # ##### comparison data [ REMOVE LATER ] #####
    # def write_comparison_data(self):
    #     bee_agents = self.agents.select(lambda agent: agent.type == 0)
    #     for a in bee_agents[3:4]:   # only log for fed bees
    #         # if a.food < 0.3:
    #         #     print("ERROR: Grabbed the wrong bee for comp data")
    #         #     print("Bee ID: " + str(a.unique_id))
    #         ## X and Y grads
    #         grads_ = [[a.grad_x, a.grad_y],[a.grad_x_old, a.grad_y_old]]
    #         self.grads_out.append(grads_)
    #         ## total Cs
    #         Cs_tmp = [a.total_C, a.total_C_old]
    #         self.Cs_out.append(Cs_tmp)

    # # write_comparison_data()
            
    # def save_comparison_data(self):
    #     df_grads = pd.DataFrame(self.grads_out, columns=['new_grad','old_grad'])
    #     # df_total_grads = pd.DataFrame(self.total_grads_out, columns=['new_total_grad','old_total_grad'])
    #     df_Cs = pd.DataFrame(self.Cs_out, columns=['new_C','old_C'])
    #     path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id)
    #     df_grads.to_csv(path + "/comparison_grads_" + str(self.run_id) + ".csv")
    #     # df_total_grads.to_csv(path + "/comparison_total_grads_" + str(self.run_id) + ".csv")
    #     df_Cs.to_csv(path + "/comparison_Cs_" + str(self.run_id) + ".csv")
    # # save_comparison_data()
    # ##### ########## #### # ###### ##### # #####

    def save_data(self):
        df = pd.DataFrame(self.data)
        # path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/run_" + str(self.run_id) + ".csv"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{self.session_id}/batch_{self.batch_id}/run_{self.run_id}.csv"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{self.session_id}/batch_{self.batch_id}/run_{self.run_id}.csv"
        df.to_csv(path)
    # save_data()

    def save_grad_data(self):
        df = pd.DataFrame(self.grad_data)
        # path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/grad_" + str(self.run_id) + ".csv"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{self.session_id}/batch_{self.batch_id}/grad_{self.run_id}.csv"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{self.session_id}/batch_{self.batch_id}/grad_{self.run_id}.csv"
        df.to_csv(path)
    # save_grad_data()

    def save_food(self):
        df = pd.DataFrame(self.food_data)
        # path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/food_" + str(self.run_id) + ".csv"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{self.session_id}/batch_{self.batch_id}/food_{self.run_id}.csv"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{self.session_id}/batch_{self.batch_id}/food_{self.run_id}.csv"
        df.to_csv(path)
    # save_food()

    def save_scent(self):
        df = pd.DataFrame(self.scent_data)
        # path = "/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/scent_" + str(self.run_id) + ".csv"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{self.session_id}/batch_{self.batch_id}/scent_{self.run_id}.csv"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{self.session_id}/batch_{self.batch_id}/scent_{self.run_id}.csv"
        df.to_csv(path)
    # save_scent()

    def write_cMap(self):
        self.environment_history.append(self.environment.concentration_map)
    # write_cMap()

    def save_cMap(self):
        # path = r"/Volumes/peleg-group-2/Richard/troph_code_data/session_" + str(self.session_id) + "/batch_" + str(self.batch_id) + "/cMap_" + str(self.run_id) + ".h5"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{self.session_id}/batch_{self.batch_id}/cMap_{self.run_id}.h5"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{self.session_id}/batch_{self.batch_id}/cMap_{self.run_id}.h5"
        with h5py.File(path, 'w') as outfile:
            outfile.create_dataset("concentration", data=self.environment_history)
    # save_cMap()

# Class TrophallaxisABM()
