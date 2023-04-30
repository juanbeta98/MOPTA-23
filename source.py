import gurobipy as gb
import networkx as nx
import numpy as np; import pandas as pd; from time import process_time
import pickle


#### parametrers ####
path = "C:/Users/a.rojasa55/OneDrive - Universidad de los Andes/Documentos/MOPTA-23/Data/"
# path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/Documentos/MOPTA-23/Data/"

#path = '/Users/juanbeta/My Drive/Research/MOPTA/MOPTA-23/Data/'

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

    file = open(path + f'/K_s/Ks_sc{scenario}', 'rb')
    K_s = pickle.load(file)
    file.close()

    file = open(path + f'/S_k/Sk_sc{scenario}', 'rb')
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

    Ks = [k for k in K if pi[f"V{k}"]>0]

    V = ["s"] + Ks + ["e"]
    A = [(i,j) for i in V for j in V if i!=j and i!="e" and j!="s" and a[i,s] < a[j,s] and (i,j)!=("s","e")]

    rc = {arc:-pi[f"V{arc[1]}"]-sigma if arc[0]=="s" else (0 if arc[1]=="e" else -pi[f"V{arc[1]}"]) for arc in A}
    
    return V,A,rc

def vertices_extensions(V,A):
    
    G = nx.DiGraph()
    G.add_nodes_from(V); G.add_edges_from(A)
    outbound_arcs = {}
    for v in V:
        outbound_arcs[v] = list(G.out_edges(v))
    
    return outbound_arcs

def label_algorithm(s,V,K,T,r,t,a,ext,pi,sigma):

    def label_extension(l,arc,dominated):
        for m in range(1):
            i = arc[0]; j = arc[1]
            new_label = [[], 0, 0, 0]

            ''' Check waiting line feasibility '''
            if a[j,s] < l[2]: break
            if a[j,s] > l[3]: new_label[2] = l[2]
            else: new_label[2] = l[3]

            ''' Check time consumption feasibility '''
            new_label[3] = l[3] + max(0,a[j,s]-l[3]) + t[j,s]
            if new_label[3] > T: break
            
            ''' Update the resources consumption '''
            new_label[0] += l[0] + [j]
            new_label[1] = l[1] + r[(i,j)]
            
            if j == "e":
                done.append(new_label)
            else:
                new_labels.append(new_label)
                label_dominance(new_label,dominated)
    
    def label_dominance(new_label,dominated):
        for l in range(len(labels)):
            if set(labels[l][0]).issubset(set(new_label[0])):
                dominated[l] = True

    ''' Labels list '''
    # Index: number of label
    # 0: route
    # 1: cumulative reduced cost
    # 2: last moment where there was a vehicle waiting in line to use the charger
    # 3: cumulative time consumption
    labels = [ [[arc[0], arc[1]], r[arc], a[arc[1],s], a[arc[1],s]+t[arc[1],s]] for arc in ext["s"]]
    done = []

    while len(labels) > 0:
        
        L = range(len(labels))
        new_labels = []
        dominated = {l:False for l in L}
        for l in L:
            for arc in ext[labels[l][0][-1]]:
                if not dominated[l]:
                    label_extension(labels[l], arc, dominated)

        del labels[:len(L)]
        labels = new_labels.copy()
    
    routes = []
    for l in range(len(done)):
        # If reduced cost is negative
        if done[l][1] < -0.001:
            col = {k:1 if k in done[l][0] else 0 for k in K}
            routes.append((col,done[l][1]+sum(pi[f"V{k}"]*col[k] for k in K)+sigma))

    return routes

def initial_routes(S,K,K_s):

    routes = {0:[]}
    for k in K:
        routes[0].append(({kk:1 if kk == k else 0 for kk in K},1))

    np.random.seed(0)
    for s in S:
        rand = np.random.choice(K_s[s])
        routes[s] = [({k:1 if k == rand else 0 for k in K_s[s]},0)]

    return routes

def master_problem(S,C,y,routes,S_c,C_s,output=0,integer=0):

    R = {s:range(len(routes[s])) for s in S+[0]}

    f_r = {s:{r:routes[s][r][1] for r in R[s]} for s in S+[0]}
    z_r = {s:{r:routes[s][r][0] for r in R[s]} for s in S+[0]}

    m = gb.Model("Restricted Master Problem")

    if integer == 1: nat = gb.GRB.BINARY
    else: nat = gb.GRB.CONTINUOUS

    lbd = {s:{r:m.addVar(name=f"lambda_{s,r}",vtype=nat) for r in R[s]} for s in S+[0]}

    for c in C:
        m.addConstr(gb.quicksum(z_r[s][r][c]*lbd[s][r] for s in S_c[c]+[0] for r in R[s]) >= 1, f"V{c}_assignment")

    for s in S:
        m.addConstr(gb.quicksum(lbd[s][r] for r in R[s]) <= y[s], f"S{s}_convexity")
    
    m.setObjective(gb.quicksum(f_r[s][r]*lbd[s][r] for s in S+[0] for r in R[s]))

    m.update()
    m.setParam("OutputFlag",output)
    m.optimize()

    z = {(c,s):sum(z_r[s][r][c]*lbd[s][r].X for r in R[s]) for s in S for c in C_s[s]}
    infeasible = [c for c in C if sum(z_r[0][r][c]*lbd[0][r].X for r in R[0]) > 0]

    pi_0 = {}
    if integer == 0:
        for c in C:
            cons = m.getConstrByName(f"V{c}_assignment")
            pi_0[f"V{c}"] = cons.getAttr("Pi")
        for s in S:
            cons = m.getConstrByName(f"S{s}_convexity")
            pi_0[f"S{s}"] = cons.getAttr("Pi")

    return pi_0,infeasible,m.getObjective().getValue(),z

def second_stage_ESPP(S,K,K_s,S_k,T,y,a,t):
    
    routes = initial_routes(S,K,K_s)
    time0 = process_time()
    i = 0; estancado = False; lastObj = 1e6
    print(f"-----Second Stage iteration {i}")
    while True:
        i += 1

        pi, infeasible, objMP,zz = master_problem(S,K,y,routes,S_k,K_s,output=0)
        print(f"Iteration {i}:\t{len(infeasible)} infeasible vehicles\tMP obj: {round(objMP,2)}\ttime: {round(process_time()-time0,2)}s")

        if lastObj - objMP < 1: estancado = True; break
        else: lastObj = objMP

        opt = {}
        for s in S:
            V,A,rc = get_graph(s,K_s[s],a,pi,pi[f"S{s}"])
            ext = vertices_extensions(V,A)
            opt[s] = label_algorithm(s,V,K,T,rc,t,a,ext,pi,pi[f"S{s}"])
            #print(f"Station {s}: {len(opt[s])} new columns")
            routes[s] += opt[s]

        if sum(len(opt[s]) for s in S) == 0: break
    
    if objMP == 0 or not estancado:
        pi, infeasible, objMP, zz = master_problem(S,K,y,routes,S_k,K_s,output=0,integer=1)

    return objMP
