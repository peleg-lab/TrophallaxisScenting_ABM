import mesa
# import solara
# from matplotlib.figure import Figure

from .model import TrophallaxisABM
from .SimpleContinuousModule import SimpleCanvas

import numpy as np

def _get_agent_color(agent):
    # if not agent.is_queen and agent.state == 2:
    #     return "rgb(0,0,200)"

    if agent.food == 0:
        return "rgb(0,0,0)"
    elif agent.food > 0.5:
        return "rgb(255,0,0)"
    elif agent.food > 0.24:
        return "rgb(255,51,51)"
    elif agent.food > 0.12:
        return "rgb(255,102,102)"
    elif agent.food > 0.06:
        return "rgb(255,153,153)"
    else:
        return "rgb(255,204,204)"
# _get_agent_color()

def bee_draw(agent):
    portrayal = {"Filled": "true"}

    ## Set shape, size, and color:
    portrayal["Shape"] = "circle"
    portrayal["r"] = 7.8
    portrayal["Color"] = _get_agent_color(agent)
    return portrayal
# bee_draw()

## Make Bee Canvas:
bee_canvas = SimpleCanvas(bee_draw, 500, 500)

## Set Model Parameters:
model_params = {
    "N": mesa.visualization.Slider(
        "N",
        110,    # default
        0,      # min
        500,    # max
        5,      # step
    ),
    # "fraction_of_fed_bees": mesa.visualization.Slider(
    #     "Fraction of fed bees",
    #     10,
    #     0,
    #     100,
    #     1,
    # ),
    # "theta": mesa.visualization.Slider(
    #     "Theta: random walk",
    #     180,
    #     0,
    #     360,
    #     5,
    # ),
    # "solid_bounds": mesa.visualization.Checkbox(
    #     "Use Solid Arena Boundaries?",
    #     True,
    # ),
###### Toggles:
    # "Dc": mesa.visualization.Slider(
    #     "Diffusion Coefficient",
    #     6.0,
    #     0.6,
    #     10.0,
    #     0.2,
    # ),
    "wb_": mesa.visualization.Slider(
        "worker bias",
        20.0,
        0.0,
        40.0,
        5.0,
    ),
    # "del_t": mesa.visualization.Slider(
    #     "delta time",
    #     0.1,
    #     0.1,
    #     1.0,
    #     0.1,
    # ),
    # "dec_rate": mesa.visualization.Slider(
    #     "Pheromone Decay Rate",
    #     5.0,
    #     0.0,
    #     20.0,
    #     1.0,
    # ),
###### Scenting Toggles:
    # "scent_thresh": mesa.visualization.Slider(
    #     "Fed Scenting Theshold",
    #     0.2,
    #     0.1,
    #     0.5,
    #     0.1
    # ),
    # "scent_prob": mesa.visualization.Slider(
    #     "Fed Scenting Probability",
    #     0.3,
    #     0.1,
    #     0.9,
    #     0.1
    # ),
    # "scent_freq": mesa.visualization.Slider(
    #     "Fed Scenting Theshold",
    #     80,
    #     5,
    #     80,
    #     5
    # ),
    # "scent_move": mesa.visualization.Checkbox(
    #     "Move during wait time?",
    #     True,
    # ),
###### Fanning Toggles:
    # "trans_prob_": mesa.visualization.Slider(
    #     "Scenting Probability",
    #     0.5,
    #     0.1,
    #     1.0,
    #     0.1,
    # ),
###### Data & Session Toggles:
    "data_out": mesa.visualization.Checkbox(
        "data csv?",
        False,
    ),
    # "food_out": mesa.visualization.Checkbox(
    #     "food csv?",
    #     False,
    # ),
    "batch_id": mesa.visualization.Slider(
        "batch number",
        0,
        0,
        10,
        1,
    ),
    "run_id": mesa.visualization.Slider(
        "run ID",
        0,
        0,
        10,
        1,
    ),
    "width": 32,
    "height": 32,
}

## Plots and Data:
foodBar = mesa.visualization.BarChartModule([
    {"Label": "Fed .50",
     "Color": "rgb(255,0,0)"},
    {"Label": "Fed .25",
     "Color": "rgb(255,51,51)"},
    {"Label": "Fed .12",
     "Color": "rgb(255,102,102)"},
    {"Label": "Fed .05",
     "Color": "rgb(255,153,153)"},
    {"Label": "Fed .01",
     "Color": "rgb(255,204,204)"},
    {"Label": "Fed 0.0",
     "Color": "rgb(0,0,0)"},
    ],
    canvas_height=300,
    canvas_width=550,
    data_collector_name="data_collector")

tmpchart = mesa.visualization.ChartModule([
    {"Label": "Fed .50",
     "Color": "Blue"}
],
    canvas_height=300,
    data_collector_name="data_collector")

# ## Custom Pheromone Plot:
# def plot_chems(model):
#     cmap = model.environment.concentration_map.copy()
#     min_c = np.min(cmap)
#     max_c = np.max(cmap)
#     fig = Figure()
#     ax = fig.subplots()
#     ax.imshow(cmap, cmap='Greens', vmin=min_c, vmax=max_c)
#     # clb = plt.colorbar(shrink=0.8, format='%.2f')
#     # clb.ax.set_title('C')
#     # plt.title("Pheromone Concentrations")
#     solara.FigureMatplotlib(fig)


## Make Server and Run:
server = mesa.visualization.ModularServer(
    TrophallaxisABM, [bee_canvas, foodBar], "Trophallaxis", model_params
)
