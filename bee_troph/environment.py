import numpy as np

class Environment(object):

    def __init__(self, x_min, x_max, dx, dec_rate=0.0, dc=0.6, del_t=0.005):
        ## Grid Values:
        self.x_min = x_min  # 0
        self.x_max = x_max  # 32
        self.dx = dx        # 0.1
        self.culling_threshold = 0.01 #0.001    # 0.0001

        ## Diffusion Values:
        self.decay_rate = dec_rate    # 5.0   # decay rate: (18.0)
        self.D = dc         # diffusion coefficient (0.6)

        ## Time Variables:
        self.t_min = 0
        self.t_max = 1000
        self.dt = del_t     # 0.1

        ## Sources:
        self.pheromone_sources = []

        ## Initialize Environment Grid:
        self.__init_environemnt_grid()      # 320x320 grid

    def __init_environemnt_grid(self):
        ## Create 320x320 grid
        X1 = np.arange(self.x_min, self.x_max+self.dx, self.dx)
        X2 = np.arange(self.x_min, self.x_max+self.dx, self.dx)
        self.x_grid, self.y_grid = np.meshgrid(X1, X2)

    #####################################
    ### Peromone Source Manipulation: ###
    #####################################

    def update_pheromone_sources(self, bee, bee_positions, t0):        # My version
        ## Add bee to sources, if emitting:
        if bee.state == 4 or bee.state == 1: #bee.type == 0:
            ## Normalize bias
            d = np.linalg.norm([bee.wx, bee.wy]) + 1e-9
            bee_tuple = {
                "bee_id"      : bee.unique_id,
                "x"       : bee_positions[f'bee_{bee.unique_id}'][0],
                "y"       : bee_positions[f'bee_{bee.unique_id}'][1],
                "x_grad"  : bee.grad_x,     # unit vector heading
                "y_grad"  : bee.grad_y,
                "wb"      : bee.wb,         # worker bias scalar = 40.0
                "wx"      : bee.wx / d,         # x bias
                "wy"      : bee.wy / d,         # y bias
                "A"       : bee.A,          # worker initial concentration
                "t_start" : t0              # starting time
            }
            ## Add identifier to list:
            self.pheromone_sources.append(bee_tuple)
    # update_pheromone_sources()

    def cull_pheromone_sources(self, t_i):
        ## Track indexes to keep:
        keep_idxs = []
        ## Iterate over sources:
        for pheromone_src_i, src in enumerate(self.pheromone_sources):
            ## Get time difference:
            delta_t = self.dt * (t_i - src['t_start'])
            delta_t += self.dt
            ## Calc diffusion:
            current_c = self._diffusion_eq(A = src['A'], D = self.D,
                                           X = (self.x_grid - src['x']),
                                           Y = (self.y_grid - src['y']),
                                           wbu = src['wb'] * src['wx'],
                                           wbv = src['wb'] * src['wy'],
                                           t = delta_t)
            ## If diffusion > threshold: Keep the index:
            if current_c.any() > self.culling_threshold:
                keep_idxs.append(pheromone_src_i)
        count = len(self.pheromone_sources) - len(keep_idxs)
        ## Remove old sources:
        self.pheromone_sources = list(np.array(self.pheromone_sources)[keep_idxs])
        return count
    # cull_pheromone_sources()

    ##########################
    ### Concentration Map: ###
    ##########################

    def init_concentration_map(self):
        ## Initialize concentration map with all zeros (no pheromones)
        self.concentration_map = np.zeros([self.x_grid.shape[0], self.x_grid.shape[0]], dtype=np.float32)
    # init_concentration_map()

    def _diffusion_eq(self, A, D, X, Y, wbu, wbv, t):     # My version
        # term_1 = (A) / (D * (t + 1e-9))
        term_1 = (A) / (D * t)
        term_2 = (X - wbu * t)**2 + (Y - wbv * t)**2
        denom = 4 * D * t
        # c = term_1 * np.exp(-(term_2 / denom) - (self.decay_rate * t))
        c = term_1 * np.exp(-(term_2 / denom))
        return c
    # _diffusion_eq()

    def update_concentration_map(self, t_i, pheromone_src):
        ## Add source pheromone to current concentration map
        delta_t = self.dt * (t_i - pheromone_src['t_start'])
        delta_t += self.dt
        current_c = self._diffusion_eq(A=pheromone_src['A'], D=self.D,
                                        X=self.x_grid - pheromone_src['x'],
                                        Y=self.y_grid - pheromone_src['y'],
                                        wbu=pheromone_src['wb']*pheromone_src['wx'],
                                        wbv=pheromone_src['wb']*pheromone_src['wy'],
                                        t=delta_t)

        self.concentration_map += current_c
        return current_c    # return current map
    # update_concentration_map()

    ### --New Version to replace above function and the loop in model.py--
    # def update_concentration_map(self, t_i):
    #     ## Iterate over pheromone sources and update concentration map:
    #     for src in self.pheromone_sources:
    #         ## Update concentration map:
    #         conc_map = self._update_source_concentration(t_i, src)
    # # update_concentration_map()
    #
    # def _update_source_concentration(self, t_i, source_agent):      # My version
    #     ## Find change in time:
    #     d_t = t_i - source_agent['t_start']
    #     d_t += self.dt
    #     ## Calc concentration map for specific agent:
    #     current_c = self._diffusion_eq(A = source_agent['A'], D = self.D,
    #                                    dx = (self.x_grid - source_agent['x']),
    #                                    dy = (self.y_grid - source_agent['y']),
    #                                    wbx = source_agent['wb'] * source_agent['wx'],
    #                                    wby = source_agent['wb'] * source_agent['wy'],
    #                                    t = d_t)
    #     ## Add new concentration map to the overall one:
    #     self.concentration_map += current_c
    #     return current_c        # return the source's concentration map
    # # _update_source_concentration()
    ### -------needs more work-------

    #############################
    ### Gradient Calculation: ###
    #############################

    def __calc_gradient(self, x_sample_pt, y_sample_pt, D, dt, A, x_source, y_source, wx, wy, wb, decay_rate):
        # K = -A / (2 * D * dt * np.sqrt(dt) + 1e-5)
        K = -A / (2 * D * D * dt + 1e-5)
        x_term = (x_sample_pt-x_source - wb*wx*dt)**2
        y_term = (y_sample_pt-y_source - wb*wy*dt)**2
        denom = dt*4*D + 1e-5
        # exp_term = np.exp(-(x_term + y_term)/denom - decay_rate*dt)
        exp_term = np.exp(-(x_term + y_term)/denom)
        dc_dx = K * exp_term * (x_sample_pt - x_source - wb*wx*dt)
        dc_dy = K * exp_term * (y_sample_pt - y_source - wb*wy*dt)
        return dc_dx, dc_dy
    # __calc_gradient()

    def calc_gradient_to_source(self, t_i, bee_x, bee_y, src):     # Mine [adaptation from calculate_gradient() function]
        ## Calculate the gradient (dx, dy) from bee_x,bee_y to specifc source
        delta_t = self.dt * (t_i - src['t_start'])
        delta_t += self.dt
        dx, dy = self.__calc_gradient(bee_x, bee_y, self.D, delta_t,
                                      src['A'],
                                      src['x'],  src['y'],
                                      src['wx'], src['wy'],
                                      src['wb'], self.decay_rate)
        return dx, dy
    # calc_gradient_to_source()

    # def calc_gradient_at_point(self, x, y, t_i):
    #     ## Calculate the total gradient at point (x,y)
    #     grad_x = 0
    #     grad_y = 0
    #     ## Iterate through sources:
    #     for src in self.pheromone_sources:
    #         ## Find gradient from source and add to total gradient:
    #         dx, dy = self.__calc_gradient_to_source(t_i, x, y, src)
    #         grad_x += dx
    #         grad_y += dy
    #     return grad_x, grad_y       # return total gradient at point (x,y)
    # # calc_gradient_at_point()

    # def calc_gradient_at_point_2(self, x, y, t_i):
    #     ## This version uses Numpy's gradient() method:
    #     dxx, dyy = np.gradient(map, 0.1)                # change to "self.concentration_map" ?
    #     dx = dxx[int(self.convert_xy_to_index(x))]
    #     dy = dyy[int(self.convert_xy_to_index(y))]
    #     return dx, dy
    # # calc_gradient_at_point_2()

    ################
    ### Helpers: ###
    ################

    def convert_xy_to_index(self, XY):
        index = ((XY - self.x_min) / (self.x_max - self.x_min)) * self.x_grid.shape[0]
        return index
    # convert_xy_to_index()

    def write_out_concentration_map_txt(self, t_i):
        file_obj = open(r"data_out/conc_map_" + str(t_i) + ".txt", "w")
        for line in self.concentration_map:
            file_obj.write(str(line) + "%\n")
        file_obj.close()
    # write_out_concentration_map_txt()

    #####################################################################
    ######### Unsure about these functions from Deiu My's code: #########

    # def __get_item(self, idx):
    #     current_t = self.t_grid[idx]
    #     self.init_concentration_map()
    #     reutnr current_t
    # # __get_item()
    #
    # def __set_params(self, params):
    #     for key, val in params.items():
    #         self.__dict__[key] = val
    # # __set_params()
    #
    # def __init_timecourse(self):
    #     print("Creating timecourse...")
    #     self.t_grid = np.arange(self.t_min, self.t_max, self.dt)
    # # __init_timecourse()



# Environment()
