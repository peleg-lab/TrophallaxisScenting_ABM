from mesa.model import Model
from bee_troph.model import TrophallaxisABM
import globals
import numpy as np

# import sys
# print("Python version: " + str(sys.version))
# print("Mesa version: " + str(sys.modules['mesa'].__version__))

import time
import os
from tqdm import tqdm

IS_LOCAL = True

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
# create_directory_if_not_exists()

def _get_model_parameters(Dc: float=0.06, wb_s: float=10.0,
                          del_t: float=0.005, dec_rate: float=7.8, cull_thresh: float=0.0001,
                          f_s_thresh: float=0.45, unfed_thresh: float=300,
                          unfed_trans_prob: float=0.2,
                          use_queens: bool=False, time_limit: int=400,
                          ses_i: int=0, b_i: int=0, r_i: int=0,
                          n: int=110, num_fed: int=10, th: int=180,):

    model_params = {
        ## Session Parameters: 
        ####  - session_id=0, 
        ####  - batch_id=0,
        ####  - run_id=0,
        "session_id": ses_i,
        "batch_id": b_i,
        "run_id": r_i,

        ## Diffusion Parameters:
        ####  - Dc=0.06, 
        ####  - wb_=20.0, 
        ####  - del_t=0.005, 
        ####  - dec_rate=5.0,
        ## diffusion coefficient: [0.01, 0.1]
        "Dc": Dc,
        ## worker bias: [0.0, 30.0]
        "wb_": wb_s,
        ## delta Time: [0.005, 0.02]
        "del_t": del_t,
        ## Decay Rate: []
        "dec_rate": dec_rate,
        ## Culling Threshold:
        "cull_thresh": cull_thresh,
        
        ## Scenting Parameters:
        ####  - fed_scent_threshold_=0.15, 
        ####  - fed_scent_prob_=0.33, 
        ####  - fed_scent_freq_=80, 
        ####  - fed_skip_wait_=True,
        "fed_scent_threshold_": f_s_thresh,   # threshold required for fed bee to emit
        "fed_scent_prob_": 0.33,      # probability for fed bee to emit
        "fed_scent_freq_": 80,       # emission frequency of fed bee
        "fed_skip_wait_": False,         # whether fed bees can move while emitting

        ## Fanning Parameters:
        ####  - unfed_threshold_=0.15, 
        ####  - unfed_trans_prob_=0.5,
        "unfed_threshold_": unfed_thresh,      # threshold for bee to fan wings and scent
        "unfed_trans_prob_": unfed_trans_prob,      ## Dieumy default value = 0.4

        ## Bee Parameters:
        ####  - N=110, 
        ####  - fraction_of_fed_bees=10, 
        ####  - theta=180,
        "N": n,
        # "fraction_of_fed_bees": frac_fed,
        "number_of_fed_bees": num_fed,
        "theta": th,

        ## Special Modes/testing options:
        ####  - use_queens_=False, 
        ####  - time_limit_=400,
        "use_queens_": use_queens,
        "time_limit_": time_limit,

        #### Arena Parameters: 
        ####  - solid_bounds=True, 
        ####  - data_out=False, 
        ####  - width=32, height=32
        "solid_bounds": True,
        "output_data_": True,
        "width": 32,
        "height": 32,
    }
    return model_params
# _get_model_parameters()

def batch_run(session_num: int = 0, batch_num: int = 0, repetitions: int = 10,
                r_start: int=0, max_steps: int = 1000, display_progress: bool = True,
                Dc: float=0.01, wb_s: float=0.0, del_t: float=0.005, 
                dec_rate: float=5.0, cull_thresh=0.0001, 
                food_thresh: float=0.15, thresh_fan: float=0.3, unfed_trans_prob: float=0.4,
                use_queens: bool=False, time_limit: int=400,
                n: int = 110, num_fed: int = 10, th: int = 180):
    ## Start terminal printout:
    if display_progress:
        print("BATCH: " + str(batch_num))

    ## Get parameters:
    ##       _get_model_parameters(D,  wb,   del_t, decay,   food_thresh, thresh_fan, si,   use_queens, time_limit,   bi, ri,   n, frac_fed, th)
    params = _get_model_parameters(Dc, wb_s, del_t, dec_rate, cull_thresh,
                                   food_thresh, thresh_fan, unfed_trans_prob,
                                   use_queens, time_limit,
                                   session_num, batch_num, r_start,
                                   n, num_fed, th)
    
    if time_limit is not None:
        max_steps = time_limit

    ## Make session directory (if necessary)
    if IS_LOCAL:
        session_path = globals.local_path_main + globals.local_path_data + f"session_{session_num}/"
    else:
        session_path = globals.scratch_path_dir + f"session_{session_num}/"
    create_directory_if_not_exists(session_path)

    ## Run model and repeat 'repetitions' times:
    id_counter = r_start                #  ---- change if not starting at run 0 ----
    for i in tqdm(range(repetitions), desc=f"Batch {batch_num} Progress"):
        start_time = time.perf_counter()
        params["run_id"] = i + id_counter

        ## Create directory for data if it doesn't exist:
        # path = f"/Volumes/peleg-group-2/Richard/troph_code_data/session_{session_num}/batch_{batch_num}"
        if IS_LOCAL:
            path = globals.local_path_main + globals.local_path_data + f"session_{session_num}/batch_{batch_num}/"
        else:
            path = globals.peleg_path_main + globals.peleg_path_data + f"session_{session_num}/batch_{batch_num}/"
        
        create_directory_if_not_exists(path)

        _run_model(TrophallaxisABM, params, max_steps)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Batch {batch_num}, Run {i + id_counter} took {elapsed_time:.6f} seconds")

    if display_progress:
        print("Batch Finished")
# batch_run()

def _run_model(model_cls: type[Model], model_params, max_steps: int):
    print("|" + str(model_params['run_id']) + "|")
    # print("|" + str(model_params['run_id']), end="|")
    model = model_cls(**model_params)
    for i in tqdm(range(max_steps), desc=f"Model Run Progress"):
    # while model.running and model.schedule.steps <= max_steps:
        model.step()

    if model.running:
        print("oops: model still running at max steps")
        model.running = False
    # print("|")
# _run_model()

####[EXAMPLE CALL]####
# if False:
#     batch_run(session_num = 0, batch_num = 0, repetitions = 10,
#               r_start=0, max_steps = 1000, display_progress = True,
#               Dc=0.01, wb_s=0.0, del_t=0.005, dec_rate=5.0,
#               thresh_scent=0.15, thresh_fan=0.3,
#               n = 110, frac_fed = 10, th = 180)

# ______________________________________________________________________
# ################### [LOG - RUNS TO PELEG SERVER] #####################
#  ____________________________________________________________________
# | s_i | b_i | runs |  Dc  |  wb  |   dt   | food_thresh | thresh_fan |
# |-----|-----|------|------|------|--------|-------------|------------|
# |  0  |  0  |  10  | .01  |  0   | 0.005  |     0.15    |    0.30    |


# batch_run(session_num = 0, batch_num = 3, repetitions = 1,
#           r_start=0, max_steps = 1000, display_progress = True,
#           Dc=0.6, wb_s=20.0, del_t=0.005, dec_rate=0.0,
#           food_thresh=0.5, thresh_fan=10.0,
#           use_queens=True, time_limit=300,
#           n = 110, num_fed = 1, th = 180)


########################################################################

# batch_run(session_num = 1, batch_num = 27, repetitions = 1,
#           r_start=0, max_steps = 1000, display_progress = True,
#           Dc=0.3, wb_s=5.0, del_t=0.05, dec_rate=0.0, cull_thresh=0.0001,
#           food_thresh=0.80, thresh_fan=0.010, unfed_trans_prob=0.2,
#           use_queens=True, time_limit=1500,
#           n = 110, num_fed = 1, th = 180)

## NOTES:
## thresh_scent should be lower, maybe 0.2
## thresh_fan should be higher, at least 100, but could be 400+

## src origin: 3333.3 -> 3349.2 -> 