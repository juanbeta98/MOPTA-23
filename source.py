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

def get_graph_chargers(s,K,a,pi,sigma):

    dist = {k:a[k,s] for k in K if pi[k]>0}
    Ks = [k for k in sorted(dist, key=dist.get)]

    V = ["s"] + Ks + ["e"]
    A = [(i,j) for i in V for j in V if i!=j and i!="e" and j!="s" and a[i,s] <= a[j,s] and (i,j) != ("s","e")]

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
        
        if v!= "s":
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
            label_DFS_exhaustive(v1,nrP,ntP,nqP,nP,cK,L,s,r,t,a,ext)


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

def label_DFS_chargers(v,rP,tP,qP,P,cK,L,s,r,t,a,ext):
    for m in range(1):
        
        if len(cK) > 0 and (P[-1] in cK or (P[-1]=="s" and v in cK)): break
        
        if v in P: break
        if a[v,s] < qP: break

        if v == "e":
            cK.update(P[1:])
            if rP < -1:
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
            label_DFS_chargers(v1,nrP,ntP,nqP,nP,cK,L,s,r,t,a,ext)
        label_DFS_chargers("e",rP,tP,nqP,nP,cK,L,s,r,t,a,ext)

def update_best_routes(L,P,rP,tP,hP,best_len,best_t,best_h,num_improv):
    L[P[0]] = (P,rP)
    best_len[P[0]] = len(P)
    best_t[P[0]] = tP
    best_h[P[0]] = hP
    num_improv[P[0]] += 1
    #print(f"\tImprov {P[0]}: {num_improv[P[0]]}, best_len = {best_len[P[0]]}")

def check_end_dominance(L,P,rP,tP,hP,best_len,best_t,best_h,num_improv,tried):
    for m in range(1):
        #print(P)
        if len(P) < best_len[P[0]]:
            tried[P[0]] += 1
            break
        elif len(P) > best_len[P[0]]:
            update_best_routes(L,P,rP,tP,hP,best_len,best_t,best_h,num_improv)
            break
        else:
            if tP < best_t[P[0]]:
                tried[P[0]] += 1
                break
            elif tP > best_t[P[0]]:
                update_best_routes(L,P,rP,tP,hP,best_len,best_t,best_h,num_improv)
                break
            else:
                if hP > best_h[P[0]]:
                    tried[P[0]] += 1
                    break
                else:
                    update_best_routes(L,P,rP,tP,hP,best_len,best_t,best_h,num_improv)
                    break

def update_max_succesors(P, marked, max_successors):
    for k in marked:
                if marked[k]:
                    ix = P.index(k)
                    if max_successors[k] == 100: max_successors[k] = len(P[ix:])
                    else:
                        max_successors[k] = np.max(( max_successors[k], len(P[ix:]) ))

def sort_extensions(v,s,tP,ext,a,t):

    if v == "s": nodes = ext
    else: nodes = ext[:-1]
    fin_times = np.array([np.max((tP,a[i,s]))+t[i,s] for i in nodes])
    sorted_indices = np.argsort(fin_times)
    sort_array = np.array(nodes)[sorted_indices].tolist()
    if v != "s": sort_array.append("e")

    return sort_array

def mod_label_DFS(v,rP,tP,qP,hP,P,cK,L,s,r,t,a,ext,best_len,best_t,best_h,num_improv,tried,marked,max_successors):
    for m in range(1):
        
        if len(P) > 1 and v!="e":
            if len(P[1:]) + max_successors[v] < best_len[P[1]]: break
            if num_improv[P[1]] > 3 or tried[P[1]] > 1000: break
        
        if cK[v]: break
        if len(P) == 1:
            li = np.concatenate([L[kk][0] for kk in L])
            cK.update({k:1 if k in li else 0 for k in cK})
            for k in cK:
                if cK[k] == 0: max_successors[k] = 100
            if cK[v]: break

        if v in P: break
        if a[v,s] < qP: break

        if v == "e":
            update_max_succesors(P, marked, max_successors)
            check_end_dominance(L,P[1:],rP,tP,hP,best_len,best_t,best_h,num_improv,tried)
            break
        nP = P + [v]

        if a[v,s] == tP - t[v,s]:
            nqP = qP
            if v != "s": marked[v] = 1
        else: nqP = tP - t[v,s]

        sort_ext = sort_extensions(v,s,tP,ext[v],a,t)
        for v1 in sort_ext:
            start_v1 = np.max((tP,a[v1,s]))
            nhP = hP + start_v1 - tP
            ntP = start_v1 + t[v1,s]
            nrP = rP + r[v,v1]
            mod_label_DFS(v1,nrP,ntP,nqP,nhP,nP,cK,L,s,r,t,a,ext,best_len,best_t,best_h,num_improv,tried,marked,max_successors)
        marked[v] = 0


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
    mp.setParam("TimeLimit",10*60)
    mp.setParam("MIPFocus",2)
    mp.update(); mp.optimize()

    return mpsol, mp.getObjective().getValue()


def second_stage_chargers(S,K,K_s,y,a,t,sc):

    mp = gb.Model(f"Restricted Master Problem_{sc}")

    dummy_0 = {k:mp.addVar(vtype=gb.GRB.CONTINUOUS, obj=1, name=f"st0_{k}_{sc}") for k in K}
    aux = {s:mp.addVar(vtype=gb.GRB.CONTINUOUS, obj=2, name=f"aux_{s}_{sc}") for s in S}

    vehic_assign = {}
    for k in K:
        vehic_assign[k] = mp.addConstr(dummy_0[k] == 1, f"V{k}_assignment_{sc}")

    st_conv = {}
    for s in S:
        st_conv[s] = mp.addConstr(aux[s] <= y[s], f"S{s}_convexity_{sc}")

    mp.setParam("OutputFlag",0)

    time0 = process_time(); i = 0
    lbd = []

    while True:
        
        mp.update()
        mp.optimize()

        pi = {k:vehic_assign[k].getAttr("Pi") for k in K}
        sigmas = {s:st_conv[s].getAttr("Pi") for s in S}

        infeasible = [k for k in K if dummy_0[k].X > 0]

        #print(f"\t\tIteration {i}:\t\tMP obj: {round(mp.getObjective().getValue(),2)}\ttime: {round(process_time()-time0,2)}s")
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
    mp.setParam("TimeLimit",10*60)
    mp.setParam("MIPFocus",2)
    mp.update(); mp.optimize()
    # print(f"ttSolved IMP scenario {sc}, with {mp.getObjective().getValue()} unnasigned vehicles")

    routes = {s:list() for s in S}
    
    for l in lbd:
        if l.X > 0.5:
            for s in S:
                if mp.getCoeff(st_conv[s],l) == 1:
                    station_route = s; break
            col = [k for k in K_s[s] if mp.getCoeff(vehic_assign[k],l) == 1]
            routes[station_route].append(col)

    return routes, sigmas