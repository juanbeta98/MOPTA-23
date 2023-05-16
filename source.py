import gurobipy as gb
import networkx as nx
import numpy as np; import pandas as pd; from time import process_time
import pickle
import os

#### parametrers ####
path = os.getcwd()+"/Data/"
#path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/Documentos/MOPTA-23/Data/"

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
    successors = {}
    for v in V:
        successors[v] = list(G.successors(v))

    return successors

def check_dominance(path,routes):
        dom = False
        for i in range(len(routes)-1,-1,-1):
            if set(path).issubset(routes[i]):
                dom = True; break
        if not dom: routes.append(path)

def label_DFS_exhaustive(v,rP,tP,qP,P,cK,L,s,r,t,a,ext):
    for m in range(1):
        
        if P[-1]=="s" and v in cK: break

        if v in P: break
        if a[v,s] < qP: break

        nP = P + [v]
        if v == "e" and rP<-0.001:
            check_dominance(P[1:],L)

        if a[v,s] == tP - t[v,s]: nqP = qP; cK.add(v)
        else: nqP = tP - t[v,s]

        for v1 in ext[v]:
            ntP = np.max((tP,a[v1,s])) + t[v1,s]
            nrP = rP + r[v,v1]
            label_DFS_exhaustive(v1,nrP,ntP,nqP,nP,L,s,r,t,a,ext)


def label_DFS(v,rP,tP,qP,P,cK,L,s,r,t,a,ext):
    for m in range(1):
        
        if len(cK) > 0 and (P[-1] in cK or (P[-1]=="s" and v in cK)): break
        
        if v in P: break
        if a[v,s] < qP: break

        if v == "e":
            cK.update(P[1:])
            if rP < -0.001:
                L.append(P[1:])
            break
        nP = P + [v]

        if a[v,s] == tP - t[v,s]: nqP = qP
        else: nqP = tP - t[v,s]

        nodes = ext[v][:-1]
        fin_times = np.array([np.max((tP,a[i,s]))+t[i,s] for i in nodes])
        sorted_indices = np.argsort(fin_times)
        sort_array = np.array(nodes)[sorted_indices].tolist()
        for v1 in sort_array:
            ntP = np.max((tP,a[v1,s])) + t[v1,s]
            nrP = rP + r[v,v1]
            label_DFS(v1,nrP,ntP,nqP,nP,cK,L,s,r,t,a,ext)
        label_DFS("e",rP,tP,nqP,nP,cK,L,s,r,t,a,ext)


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

        print(f"\t\tIteration {i}:\t\tMP obj: {round(mp.getObjective().getValue(),2)}\ttime: {round(process_time()-time0,2)}s")
        i += 1

        opt = 0
        for s in S:
            
            V,A,rc = get_graph(s,K_s[s],a,pi,sigmas[s])
            ext = vertices_extensions(V,A)

            #opt[s] = label_algorithm(s,K,K_s[s],T,rc,t,a,ext,mp,lbd,vehic_assign,st_conv)

            routes_DFS = []; covered_nodes = set()
            label_DFS(v="s",rP=0,tP=0,qP=0,P=[],cK=covered_nodes,L=routes_DFS,s=s,r=rc,t=t,a=a,ext=ext)
            opt += len(routes_DFS)

            for l in routes_DFS:
                col = {k:1 if k in l else 0 for k in K}
                new_Col = gb.Column(col.values(),vehic_assign.values())
                lbd.append(mp.addVar(vtype=gb.GRB.CONTINUOUS,obj=0,ub=1,column=new_Col))

                mp.chgCoeff(st_conv[s],lbd[-1],1)

        if opt == 0: mpsol = mp.getObjective().getValue(); break
    
    for v in mp.getVars():
        v.vtype = gb.GRB.BINARY
    if mpsol > 0: mp.setParam("MIPGap",0.01)
    mp.setParam("MIPFocus",2)
    mp.update(); mp.optimize()

    return mpsol, mp.getObjective().getValue()
