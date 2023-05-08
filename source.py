import gurobipy as gb
import networkx as nx
import numpy as np; import pandas as pd; from time import process_time
import pickle


#### parametrers ####
path = "C:/Users/a.rojasa55/OneDrive - Universidad de los Andes/Documentos/MOPTA-23/Data/"

vehicles = pd.read_csv(path+'MOPTA2023_car_locations.csv', sep = ',', header = None)

stations = pd.read_csv(path+"fuel_stations.csv")

northern = (-79.761960, 42.269385)
southern = (-76.9909,39.7198)
western = (-80.519400, 40.639400)
eastern = (-74.689603, 41.363559)

stations_loc = stations[["Longitude","Latitude"]]
stations_loc["Latitude"] = (stations["Latitude"]-southern[1])*69*165/178
stations_loc["Longitude"] = (stations["Longitude"]-western[0])*53

stations = stations_loc[(stations_loc["Longitude"] <= 290) & (stations_loc["Latitude"] <= 150)]
stations.rename(columns={"Longitude": 0, "Latitude":1}, inplace=True)
#### parametrers ####

def load_pickle(path, scenario):

    file = open(path + f'/K/K_sc{scenario}', 'rb')
    K = pickle.load(file)
    file.close()

    file = open(path + f'/K_s10/Ks_sc{scenario}', 'rb')
    K_s = pickle.load(file)
    file.close()

    file = open(path + f'/S_k10/Sk_sc{scenario}', 'rb')
    S_k = pickle.load(file)
    file.close()
    
    file = open(path + f'/a t/at_{scenario}', 'rb')
    a,t = pickle.load(file)
    file.close()

    return K, K_s, S_k, a, t

def first_stage(S, c_f):

    model = gb.Model('first stage')
    N = range(1,9)

    x = dict()
    x.update({(n,s): model.addVar(vtype=gb.GRB.BINARY, name=f'x_{n}{s}') for n in N for s in S})

    model.addConstr(gb.quicksum(x[n,s] for n in N for s in S) == 600)

    for s in S:
        model.addConstr(gb.quicksum(x[n,s] for n in N) <= 1)
    
    model.setObjective(gb.quicksum(c_f*n*x[n,s] for n in N for s in S))

    model.update()
    model.setParam('OutputFlag',0)
    model.optimize()

    return {s:sum(n*x[n,s].X for n in N) for s in S if sum(x[n,s].X for n in N) == 1}, model, x

def get_graph(s,K,a,pi,sigma):

    dist = {k:a[k,s] for k in K if pi[k]>0}
    Ks = [k for k in sorted(dist, key=dist.get)]

    V = ["s"] + Ks + ["e"]
    A = [(i,j) for i in V for j in V if i!=j and i!="e" and j!="s" and a[i,s] <= a[j,s] and (i,j)!=("s","e")]

    rc = {arc:-pi[arc[1]]-sigma if arc[0]=="s" else (0 if arc[1]=="e" else -pi[arc[1]]) for arc in A}
    
    return V,A,rc

def vertices_extensions(V,A):
    
    G = nx.DiGraph()
    G.add_nodes_from(V); G.add_edges_from(A)
    outbound_arcs = {}
    for v in V:
        outbound_arcs[v] = list(G.out_edges(v))
    
    return outbound_arcs

def label_algorithm(s,K,K_s,T,r,t,a,ext,mp,lbd,vehic_constr,stat_constr):

    def label_extension(l,arc,dominated):
        for m in range(1):
            i = arc[0]; j = arc[1]
            new_label = [[], 0, 0, 0]

            ''' Check cycle feasibility '''
            if j in l[0]: break

            ''' Check waiting line feasibility '''
            if a[j,s] < l[2]: break
            if a[j,s] > l[3]: new_label[2] = l[2]
            else: new_label[2] = l[3]

            ''' Check time consumption feasibility '''
            #if l[3] > a[j,s] + T: break
            new_label[3] = l[3] + max(0,a[j,s]-l[3]) + t[j,s]
            
            ''' Update the resources consumption '''
            new_label[0] += l[0] + [j]
            new_label[1] = l[1] + r[(i,j)]
            
            if j == "e":
                done.append(new_label)
            else:
                new_labels[j].append(new_label)
                label_dominance(new_label,j,dominated)
    
    def label_dominance(new_label,j,dominated):
        for l in range(len(labels[j])):
            if set(labels[j][l][0]).issubset(set(new_label[0])):
                dominated[j][l] = True

    ''' Labels list '''
    # Index: number of label
    # 0: route
    # 1: cumulative reduced cost
    # 2: last moment where there was a vehicle waiting in line to use the charger
    # 3: cumulative time consumption
    labels = dict()
    for k in K_s:
        if ("s",k) in ext["s"]: labels[k] = [ [["s",k], r["s",k], a[k,s], a[k,s]+t[k,s]] ]
        else: labels[k] = []
    done = []

    act = 1
    while act > 0:
        
        L = {k:len(labels[k]) for k in K_s}
        new_labels = {k:[] for k in K_s}
        dominated = {k:{l:False for l in range(L[k])} for k in K_s}
        for k in K_s:
            for l in range(L[k]):
                if not dominated[k][l]:
                    for arc in ext[labels[k][l][0][-1]]:
                        label_extension(labels[k][l], arc, dominated)

        labels = new_labels.copy()
        act = sum(len(labels[k]) for k in K_s)
    
    routes = 0
    for l in range(len(done)):
        # If reduced cost is negative
        if done[l][1] < -0.001:
            col = {k:1 if k in done[l][0] else 0 for k in K}
            new_Col = gb.Column(col.values(),vehic_constr.values())
            lbd.append(mp.addVar(vtype=gb.GRB.CONTINUOUS,obj=0,column=new_Col))

            mp.chgCoeff(stat_constr[s],lbd[-1],1)

            routes += 1

    return routes


def label_DFS(v,rP,tP,qP,P,cK,L,s,r,t,a,T,ext):
    for m in range(1):
        
        if len(cK) > 0 and (P[-1] in cK or (P[-1]=="s" and v in cK)): break
        
        if v in P: break
        if a[v,s] < qP: break
        if tP - t[v,s] > a[v,s] + T: break

        nP = P + [v]
        if v == "e":
            cK.update(nP[1:-1])
            if rP < -0.001:
                L.append(P[1:])

        if a[v,s] == tP - t[v,s]: nqP = qP
        else: nqP = tP - t[v,s]

        for arc in ext[v]:
            vv = arc[1]
            ntP = max(tP,a[vv,s]) + t[vv,s]
            nrP = rP + r[v,vv]
            label_DFS(vv,nrP,ntP,nqP,nP,cK,L,s,r,t,a,T,ext)


def second_stage_ESPP(S,K,K_s,S_k,T,y,a,t):

    mp = gb.Model("Restricted Master Problem")

    dummy_0 = {k:mp.addVar(vtype=gb.GRB.CONTINUOUS, obj=1, name=f"st0_{k}") for k in K}
    aux = {s:mp.addVar(vtype=gb.GRB.CONTINUOUS, obj=2, name=f"aux_{s}") for s in S}

    vehic_assign = {}
    for k in K:
        vehic_assign[k] = mp.addConstr(dummy_0[k] >= 1, f"V{k}_assignment")

    st_conv = {}
    for s in S:
        st_conv[s] = mp.addConstr(aux[s] <= y[s], f"S{s}_convexity")

    mp.setParam("OutputFlag",0)

    time0 = process_time(); i = 0
    lbd = []

    while True:
        
        mp.update()
        mp.optimize()

        pi = {k:vehic_assign[k].getAttr("Pi") for k in K}
        sigmas = {s:st_conv[s].getAttr("Pi") for s in S}

        infeasible = [k for k in K if dummy_0[k].X > 0]

        print(f"\t\tIteration {i}:\t{len(infeasible)} infeasible vehicles\tMP obj: {round(mp.getObjective().getValue(),2)}\ttime: {round(process_time()-time0,2)}s")
        i += 1

        opt = 0
        for s in S:
            
            V,A,rc = get_graph(s,K_s[s],a,pi,sigmas[s])
            ext = vertices_extensions(V,A)

            #opt[s] = label_algorithm(s,K,K_s[s],T,rc,t,a,ext,mp,lbd,vehic_assign,st_conv)

            routes_DFS = []; covered_nodes = set()
            label_DFS(v="s",rP=0,tP=0,qP=0,P=[],cK=covered_nodes,L=routes_DFS,s=s,r=rc,t=t,a=a,T=T,ext=ext)
            opt += len(routes_DFS)

            for l in routes_DFS:
                col = {k:1 if k in l else 0 for k in K}
                new_Col = gb.Column(col.values(),vehic_assign.values())
                lbd.append(mp.addVar(vtype=gb.GRB.CONTINUOUS,obj=0,ub=1,column=new_Col))

                mp.chgCoeff(st_conv[s],lbd[-1],1)

        if opt == 0: break
    
    if mp.getObjective().getValue() == 0:
        for v in mp.getVars():
            v.vtype = gb.GRB.BINARY
        mp.update(); mp.optimize()

    return mp.getObjective().getValue(), infeasible
