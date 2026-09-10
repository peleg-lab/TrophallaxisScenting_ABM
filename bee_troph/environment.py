import numpy as np

class Environment(object):

    def __init__(self, x_min, x_max, dx, dec_rate=0.0, dc=0.6, del_t=0.005, cull_thresh=0.0001):
        ## Grid Values:
        self.x_min = x_min  # 0
        self.x_max = x_max  # 32
        self.dx = dx        # 0.1

        ## Culling Threshold:
        #  - 0.0001 default
        #  - increase to 0.0005 (want to make sim run faster)
        self.culling_threshold = cull_thresh #0.0001 #0.5 #0.001    # 0.0001

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
            # if current_c.any() > self.culling_threshold:
            if True in (current_c > self.culling_threshold) or (src['bee_id'] == 0 and (t_i - src['t_start'] < 80)):   # (add any other conditions to keep specific bees' pheromones)
                keep_idxs.append(pheromone_src_i)
            # else:
            #     print(f" - Culled pheromone: {np.max(current_c)}")# {np.max(current_c)}")
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
        c = term_1 * np.exp(-(term_2 / denom) - (self.decay_rate * t))
        # c = term_1 * np.exp(-(term_2 / denom))
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

    # def calc_gradient_at_point(self, x, y):
    #     """
    #     Return the spatial gradient (dC/dx, dC/dy) of the current
    #     `self.concentration_map` at physical coordinates (x, y).

    #     Uses a central-difference computed via `np.gradient` to build
    #     gradient fields on the grid, then bilinearly interpolates the
    #     gradient fields to the requested (x,y) point.

    #     Returns (grad_x, grad_y).
    #     """
    #     if not hasattr(self, 'concentration_map'):
    #         raise AttributeError('concentration_map not initialized; call init_concentration_map() first')

    #     # Compute gradient fields. np.gradient returns gradient along
    #     # axis 0 (rows / y) then axis 1 (cols / x). Provide spacing
    #     # equal to self.dx for both axes to get spatial derivatives.
    #     grad_y_field, grad_x_field = np.gradient(self.concentration_map, self.dx, self.dx)

    #     nrows, ncols = self.concentration_map.shape

    #     ## scale x,y to match grid indices (0.1 spacing)
    #     x = x*10.0
    #     y = y*10.0

    #     # Convert physical coordinates to fractional array indices
    #     # row corresponds to y, col corresponds to x
    #     row_f = (y - self.x_min) / self.dx
    #     col_f = (x - self.x_min) / self.dx

    #     # Clamp to valid range
    #     row_f = np.clip(row_f, 0.0, float(nrows - 1))
    #     col_f = np.clip(col_f, 0.0, float(ncols - 1))

    #     i0 = int(np.floor(row_f))
    #     j0 = int(np.floor(col_f))
    #     i1 = min(i0 + 1, nrows - 1)
    #     j1 = min(j0 + 1, ncols - 1)

    #     wy = row_f - i0
    #     wx = col_f - j0

    #     # Bilinear interpolation helper
    #     def _bilinear(field):
    #         return (
    #             (1 - wy) * (1 - wx) * field[i0, j0]
    #             + (1 - wy) * wx * field[i0, j1]
    #             + wy * (1 - wx) * field[i1, j0]
    #             + wy * wx * field[i1, j1]
    #         )

    #     grad_x = _bilinear(grad_x_field)
    #     grad_y = _bilinear(grad_y_field)

    #     return grad_x, grad_y
    # # calc_gradient_at_point()

    def calc_gradient_at_point_2(self, x, y):
        grads_y, grads_x = np.gradient(self.concentration_map, self.dx, self.dx)
        gx2 = grads_x[round(y*10.0), round(x*10.0)]
        gy2 = grads_y[round(y*10.0), round(x*10.0)]
        return gx2, gy2
    # calc_gradient_2()

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
