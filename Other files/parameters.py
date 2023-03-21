#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from gurobipy import *
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import truncnorm, bernoulli # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

def upload_data(path):
    if path == 'Juan':
        path = "/Users/juanbeta/Library/CloudStorage/OneDrive-UniversidaddelosAndes/MOPTA 23/Data/"
    elif path == 'Ari_p':
        path = ''
    elif path == 'Ari_u':
        path = "C:/Users/jp.rodriguezr2/Universidad de los Andes/Juan Manuel Betancourt Osorio - MOPTA '23/Data/"
    coor = pd.read_csv(path + 'MOPTA2023_car_locations.csv', sep = ',', header = None)
    return coor

path = input('File path to csv')
coor = upload_data(path)

#%% Simulation functions
# Generate random vehicles' range
def generate_range() -> float:
    loc, scale, min_v, max_v = 100, 50, 20, 250
    a, b = (min_v - loc) / scale, (max_v - loc) / scale
    realization = truncnorm.rvs(a = a, b = b, loc = loc, scale = scale, size = 1)
    return realization[0]

# Desire to visit charging station (x = range)
def visit_p(x: float) -> float:
    lbda = 0.012
    return np.exp(-lbda**2*(x-20)**2)

def charging_realization() -> list:
    p = visit_p(generate_range())
    return bernoulli(p).rvs()

#%% Parameters
# Costs
c_f = 5000 + 500
c_d = 0.041 # ($/mile)
c_w = 0.0388 # ($/mile)

# Rates
driving_speed = 75 # miles per hour
steps_per_hour = 6 # each time step is ten minutes
charging_speed = 75*12/7 # miles per hour

# Number of chargers
u = 8
q = 2

#%% Distance function

def distance_matrix(df1, df2):
    # Extract the x and y coordinates as arrays
    x1 = np.array(df1[0])
    y1 = np.array(df1[1])
    x2 = np.array(df2[0])
    y2 = np.array(df2[1])
    
    # Compute the pairwise distances between the two sets of points
    distances = cdist(np.column_stack((x1, y1)), np.column_stack((x2, y2)))
    
    # Convert the distance matrix to a dictionary
    distance_dict = {}
    for i in range(len(df1)):
        for j in range(len(df2)):
            location1 = i
            location2 = j
            distance = distances[i,j]
            distance_dict[(location1, location2)] = round(distance,2)
            
    return distance_dict