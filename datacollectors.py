## DataCollectors
import mesa
# import numpy as np
# import math
# import random as rand


def num_fed_50(model):
    cn = 0
    for a in model.schedule.agents:
        if a.type == 0 and a.food >= 0.5:
            cn += 1
    return cn

def num_fed_25(model):
    cn = 0
    for a in model.schedule.agents:
        if a.type == 0 and a.food >= 0.25 and a.food < 0.5:
            cn += 1
    return cn

def num_fed_12(model):
    cn = 0
    for a in model.schedule.agents:
        if a.type == 0 and a.food >= 0.12 and a.food < 0.25:
            cn += 1
    return cn

def num_fed_05(model):
    cn = 0
    for a in model.schedule.agents:
        if a.type == 0 and a.food >= 0.05 and a.food < 0.12:
            cn += 1
    return cn

def num_fed_01(model):
    cn = 0
    for a in model.schedule.agents:
        if a.type == 0 and a.food >= 0.01 and a.food < 0.05:
            cn += 1
    return cn

def num_fed_00(model):
    cn = 0
    for a in model.schedule.agents:
        if a.type == 0 and a.food <= 0.01:
            cn += 1
    return cn
