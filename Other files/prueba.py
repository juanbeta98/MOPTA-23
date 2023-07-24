import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import gurobipy as gb

from time import process_time
from source import *

file = open(path + f'S', 'rb')
global_S = pickle.load(file); cf = 5500; N = range(1,9); T = 2.5
file.close()

y, model, x = first_stage(global_S,cf)
S = list(y.keys())

time0 = process_time()
optimal = False; ii = 0
while not optimal:

    print(f" First Stage Iteration {ii}: {process_time()-time0}")
    K, K_s, S_k, a, t = load_pickle(path,0)
    S = [s for s in S if K_s[s]!=[]]

    S_k = {k:[s for s in S_k[k] if s in S] for k in K}
    a.update({("s",s):0 for s in S})
    a.update({("e",s):T for s in S})
    t.update({("e",s):0 for s in S})

    obj = second_stage_ESPP(S,K,K_s,S_k,T,y,a,t)

    if obj == 0:
        optimal = True
    else:
        model.addConstr(gb.quicksum(x[n,s] for n in N for s in global_S if x[n,s].X < 0.5) >= 1)
        model.update()
        model.optimize()

        y = {s:sum(n*x[n,s].X for n in N) for s in global_S if sum(x[n,s].X for n in N) == 1}
        S = list(y.keys())
        ii += 1
