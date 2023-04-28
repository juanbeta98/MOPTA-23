from gurobi import *


def first_stage(S, c_f):

    model = Model('first stage')
    N = range(1,9)

    x = dict()
    x.update({(n,s): model.addVar(vtype=GRB.BINARY, name=f'x_{n}{s}')} for n in N for s in S)

    model.addConstr(quicksum(x[n,s] for n in N for s in S) == 600)

    for s in S:
        model.addConstr(quicksum(x[n,s] for n in N) <= 1)
    
    model.setObjective(quicksum(c_f*n*x[n,s] for n in N for s in S))

    model.update()
    model.setParam('OutputFlag',0)
    model.optimize()
    