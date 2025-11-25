from mesa.model import Model
from bee_troph.model import TrophallaxisABM
import numpy as np

import sys
print(sys.version)

def _get_model_parameters(Dc: float=0.06, wb_s: float=10.0,
                          del_t: float=0.005, dec_rate: float=7.8,
                          thresh_s: float=0.45, thresh_f: float=300,
                          ses_i: int=0, b_i: int=0, r_i: int=0,
                          n: int=110, frac_fed: int=10, th: int=180,):

    model_params = {
        #### Session Parameters: session_id=0, batch_id=0, run_id=0,
        "session_id": ses_i,
        "batch_id": b_i,
        "run_id": r_i,

        #### Diffusion Parameters: A_=0.575, Dc=0.01, wb_=40.0, del_t=0.005, dec_rate=5.0,
        ## Initial Concentration: [1.0]
        "A_": 1.0,
        ## Diffusion Coefficient: [0.01, 0.1]
        "Dc": Dc,
        ## worker bias: [0.0, 30.0]
        "wb_": wb_s,
        ## delta Time: [0.005, 0.02]
        "del_t": del_t,
        ## Decay Rate: []
        "dec_rate": dec_rate,

        #### Scenting Parameters: scent_thresh=0.15, scent_prob=0.5, scent_freq=80, scent_move=True,
        "scent_thresh": thresh_s,   # threshold required for fed bee to emit
        "scent_prob": 0.33,      # probability for fed bee to emit
        "scent_freq": 80,       # emission frequency of fed bee
        "scent_move": True,         # whether fed bees can move while emitting

        #### Fanning Parameters: fan_threshold=0.1,trans_prob_=0.5,
        "fan_threshold": thresh_f,      # threshold for bee to fan wings and scent
        "trans_prob_": 0.5,

        #### Bee Parameters: N=110, fraction_of_fed_bees=10, theta=180,
        "N": n,
        "fraction_of_fed_bees": 1.0/n,#frac_fed,
        "theta": th,

        #### Arena Parameters: solid_bounds=True, data_out=False, width=32, height=32
        "solid_bounds": True,
        "data_out": True,
        "width": 32,
        "height": 32,
    }
    return model_params
# _get_model_parameters()

def batch_run(session_num: int = 0, batch_num: int = 0, repetitions: int = 10,
                r_start: int=0, max_steps: int = 1000, display_progress: bool = True,
                Dc: float=0.01, wb_s: float=0.0, del_t: float=0.005, dec_rate: float=5.0,
                thresh_scent: float=0.15, thresh_fan: float=0.3,
                n: int = 110, frac_fed: int = 10, th: int = 180):
    ## Start terminal printout:
    if display_progress:
        print("BATCH: " + str(batch_num))

    ## Get parameters:
    ##       _get_model_parameters(D,  wb,   del_t, decay,    thresh_s,     thresh_fan, si,          bi,        ri,      n, frac_fed, th)
    params = _get_model_parameters(Dc, wb_s, del_t, dec_rate, thresh_scent, thresh_fan, session_num, batch_num, r_start)

    ## Run model and repeat 'repetitions' times:
    id_counter = r_start                #  ---- change if not starting at run 0 ----
    for i in range(repetitions):
        params["run_id"] = i + id_counter
        _run_model(TrophallaxisABM, params, max_steps)
        # if display_progress:
        # if display_progress:
        #     if i%5 == 0:
        #         # print("- " + str(i+1) + " of " + str(repetitions))
        #         print(str(int((i+1)/repetitions*100))+"%")
    if display_progress:
        print("Batch Finished")
# batch_run()# # # # # # # # # #

def _run_model(model_cls: type[Model], model_params, max_steps: int): #, iter: int, rep: int, max_reps: int):
    print("|" + str(model_params['run_id']), end="|")
    model = model_cls(**model_params)
    while model.running and model.schedule.steps <= max_steps:
        model.step()
        if model.t_i%30 == 0:
            print("#",end="")
    if model.running:
        print("oops")
    print("|")
# _run_model()

####[EXAMPLE CALL]####
# if False:
#     batch_run(session_num = 0, batch_num = 0, repetitions = 10,
#               r_start=0, max_steps = 1000, display_progress = True,
#               Dc=0.01, wb_s=0.0, del_t=0.005, dec_rate=5.0,
#               thresh_scent=0.15, thresh_fan=0.3,
#               n = 110, frac_fed = 10, th = 180)

##############[RUNS TO PELEG SERVER]##############
### -SESSION 0- ###

# | s_i | b_i | runs |  Dc  |  wb  |   dt   | thresh_scent | thresh_fan |
# |  0  |  0  |   1  |  .05 |  15  |  0.005 |     0.15     |    0.15    |
# |  1  |  0  |   1  |  .05 |  10  |  0.005 |     0.05     |    0.05    |
# |  1  |  1  |   1  |  .05 |  10  |  0.005 |     0.30     |    0.30    |
# |  1  |  2  |   1  |  .05 |  20  |  0.005 |     0.10     |    0.10    |
# |  1  |  3  |   1  |  .05 |  25  |  0.005 |     0.10     |    0.10    |
# |  1  |  4  |   1  |  .1  |  15  |  0.005 |     0.10     |    0.10    |

# |  1  |  5  |   1  |  .06 |  10  |  0.005 |     0.50     |    0.10    |
# |  1  |  6  |   1  |  .06 |  15  |  0.005 |     0.50     |    0.10    |
# |  1  |  7  |   1  |  .06 |  20  |  0.005 |     0.50     |    0.10    |

batch_run(session_num = 1, batch_num = 0, repetitions = 1,
          r_start=0, max_steps = 1000, display_progress = True,
          Dc=0.06, wb_s=15.0, del_t=0.005, dec_rate=0.0,
          thresh_scent=0.2, thresh_fan=100,
          n = 110, frac_fed = 10, th = 180)

# batch_run(session_num = 1, batch_num = 4, repetitions = 1,
#           r_start=0, max_steps = 1000, display_progress = True,
#           Dc=0.06, wb_s=20.0, del_t=0.005, dec_rate=0.0,
#           thresh_scent=0.50, thresh_fan=0.10,
#           n = 110, frac_fed = 10, th = 180)
#
# batch_run(session_num = 1, batch_num = 5, repetitions = 1,
#           r_start=0, max_steps = 1000, display_progress = True,
#           Dc=0.06, wb_s=20.0, del_t=0.005, dec_rate=0.0,
#           thresh_scent=0.50, thresh_fan=0.10,
#           n = 110, frac_fed = 10, th = 180)

## NOTES:
## thresh_scent should be lower, maybe 0.2
## thresh_fan should be higher, at least 100, but could be 400+
