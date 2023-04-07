# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:39:05 2021
Experiments (running time)
@author:The optimistics
"""
from gurobipy import *
import numpy as np
import math
import time
import sys
import networkx as nx
import folium
import random
import scipy.stats

COLORS = ['blue', 'green', 'yellow', 'pink', 'gold', 'violet', 'orange', 'magenta', 'cyan', 'lightcoral', 'skyblue', 'steelblue', 'lightgreen', 'indianred', 'palegreen', 'darkmagenta', 'mediumblue', 'slategrey', 'olive', 'pink']
n = int(sys.argv[1]) 
#n = 20
N = list(range(1,n+1))          #set o nodes
V = list(range(0,n+2))          #set of vertices
M = list(range(1,n+1))                                               #set of teams. |M|>>0. |M| = |N| 
location = {}                                                        #node's location
decimals = 1                                                         #precision
L = 250                                                              #latest time
c_f = 100                                                           #hiring cost
c_t = 1                                                              #unit travel time cost
c_o = 2                                                              #unit overtime cost
alpha = 0.5

t_max_lists = [10, 300, 1800, 3600, 10000]

######## customer's mean service team ########
mean_service_time = 45
#std_service_time = 0.5*mean_service_time
    

# Subproblem
def subProblem(pi):
    
    c_mod = calculateModifiedCost(pi)
    #Shortest path
    sp = Model('SP')
    sp.Params.OutputFlag = 0

    #variables
    x = {(i,j): sp.addVar(vtype = GRB.BINARY, lb = 0, ub = 1, obj = c_mod[(i,j)], name = f"x_{i},{j}") for (i,j) in A}
    Delta = sp.addVar(vtype = GRB.CONTINUOUS, lb = 0, obj = c_o, name = "Delta")
    
    #constraints
    for i in N:
        sp.addConstr(sum(x[i,j] for j in N+[n+1] if i!=j) - sum(x[j,i] for j in N+[0] if i!=j) == 0)
    
    sp.addConstr(sum(x[0,j] for j in N) == 1)
    sp.addConstr(sum(x[i,n+1] for i in N) == 1)
    
    for i in N:
        sp.addConstr(sum(x[i,j] for j in N + [n+1] if i!=j)<=1)
    
    #2-cut constraints
    for i in N:
        for j in N:
            if i!=j:
                sp.addConstr(x[i,j]+x[j,i]<=1)
    
    
    #Time limit
    sp.addConstr(sum(t_tilde[i,j]*x[i,j] for (i,j) in A)<=L+Delta)
    
    sp.modelSense = GRB.MINIMIZE
    #sp.write("SP.lp")
    sp.update()
    sp.optimize()
    
    #print("Eliminating subtours")
    arcs_aux = {(i,j):x[i,j].x for (i,j) in A if x[(i,j)].x>0.1}
    arcs = {i:j for (i,j) in A if x[(i,j)].x>0.1}
    (subtours, dummy_subtour) = findSubtours(arcs)
    elementary_dummy_path = recoverDummySubtour(dummy_subtour)
    (rc_dummy, cost_dummy, time_dummy) = calculateRouteReducedCost(elementary_dummy_path, pi)
    import time 
    #while len(subtours)>0 and rc_dummy>=-0.05 and (time.time()-initialization_time_stamp)<=t_max:
    while len(subtours)>0 and rc_dummy>=-0.05:
        for tour in subtours:
            #sp.addConstr(sum(x[i,j] for i in tour for j in tour if i!=j)<= len(tour)-1)
            for k in tour:
                sp.addConstr(sum(x[i,j] for i in tour for j in tour if i!=j)<=sum(x[i,j] for i in tour for j in N+[n+1] if i!=k and j!=i))
            
        sp.optimize()
        #sp.write("SP.lp")
        arcs = {i:j for (i,j) in A if x[(i,j)].x>0.1}
        (subtours, dummy_subtour) = findSubtours(arcs)
        elementary_dummy_path = recoverDummySubtour(dummy_subtour)
        (rc_dummy, cost_dummy, time_dummy) = calculateRouteReducedCost(elementary_dummy_path, pi)
    
    #Recover elementary path
    if rc_dummy<-0.005:
        elementary_path = elementary_dummy_path
        rc = rc_dummy
        cost = cost_dummy
        time = time_dummy
    else:
        elementary_path = [0, arcs[0]]
        nextNode = arcs[0]
        while nextNode!=n+1:
            nextNode = arcs[nextNode]
            elementary_path.append(nextNode)
    (rc,cost, time) = calculateRouteReducedCost(elementary_path, pi)
        
    return(elementary_path,rc, cost)

# Subproblem
def subProblemHeuristic(pi):
    
    c_mod = calculateModifiedCost(pi)
    G = nx.Graph()
    
    #nodes
    #only nodes with dual variable greater than zero
    for i in pi.keys():
        if pi[i]>0.1:
            G.add_node(i)
    G.add_node(0)
    
    #edges
    #start
    for j in G.nodes():
        if j!=0:
            G.add_edge(0,j, weight = c_mod[(0,j)])
    
    #intermediate
    for i in G.nodes():
        for j in G.nodes():
            if i!=j and i!=0 and j!=0:
                G.add_edge(i,j, weight = min(c_mod[(i,j)],c_mod[(j,i)]))
    
    mst = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="prim")
    edgelist = list(mst)
    
    #Preorder of the tree
    edgelist
    G_tree = nx.Graph()
    G_tree.add_edges_from(edgelist)
    G_tree.add_nodes_from(G.nodes)
    route = list(nx.dfs_preorder_nodes(G_tree, source=0, depth_limit=len(N)))
    route.append(n+1)    
    
    ######### Split (Shortest path - DAG) ################
    #Split: Not necessary. 
    #Every route with negative reduced cost should be added ()
    giantTour = route[0:-1]
    
    G_dag =nx.DiGraph()
    G_dag.add_nodes_from(giantTour)
    
    #Information
    elementary_paths = []
    rcs = []
    costs = []
   
    
    for i in range(len(giantTour)):
        for j in range(i+1, len(giantTour)):
            if i == 0:
                route = giantTour[i:j+1]+[n+1]
                (route_rc, route_cost, route_time) = calculateRouteReducedCost(giantTour[i:j+1]+[n+1], pi)
            else:
                route = [0]+giantTour[i+1:j+1]+[n+1]
                (route_rc, route_cost, route_time) = calculateRouteReducedCost([0]+giantTour[i+1:j+1]+[n+1], pi)
            G_dag.add_edge(giantTour[i], giantTour[j], weight = route_cost)
    
            """
            if route_rc<0:
                elementary_paths.append(route)
                rcs.append(route_rc)
                costs.append(route_cost)
            """
    bf = nx.bellman_ford_path(G_dag, source = 0, target = giantTour[-1])
    
    index = 1
    until_node = bf[index]
    path =[0]
    for i in giantTour[1:]:
        path.append(i)
        if i == until_node:
            if index<len(bf)-1:
                index+=1
                until_node = bf[index]
            path.append(n+1)
            route = path.copy()
            (route_rc, route_cost, route_time) = calculateRouteReducedCost(route, pi)
            if route_rc<0:
                elementary_paths.append(route)
                rcs.append(route_rc)
                costs.append(route_cost)
            path = [0]
    return(elementary_paths, rcs, costs)   


def findSubtours(arcs):
    "function to find the subtours in the shortest path problem solution"
    arcs[n+1] = 0 #dummy cycle
    check = {key: False for key in arcs.keys()}
    subtours = []
    dummy_subtour = []
    for i in arcs.keys():
        if not check[i]:
            dummyCycle = False
            node_tour = [i]
            nextNode = arcs[i]
            while nextNode!=i:
                node_tour.append(nextNode)
                nextNode = arcs[nextNode]
                if nextNode == 0 or nextNode == n+1:
                    dummyCycle = True
            if not dummyCycle:
                subtours.append(node_tour)
            else:
                dummy_subtour = node_tour
            for node in node_tour:
                check[node] = True
    return (subtours, dummy_subtour)
        
        

# Column generation loop
def columnGeneration():
    """
    This function performs a Column Generation procedure
    """
    print("Entering function columnGeneration.",flush = True)
    card_omega = len(y)
    import time  
    #while (time.time()-initialization_time_stamp)<=t_max:
    while True:
        print("t=", (time.time()-initialization_time_stamp)," t_max=", t_max, sep = "", end ="\n")
        print('Solving Master Problem (MP)...', flush = True)
        modelMP.optimize()
        print('Value of LP relaxation of MP: ', modelMP.getObjective().getValue(), flush = True)
        #Retrieving duals of master problem
        pi = {i:ctr.pi for i,ctr in NodeCtr.items()}

        #Solve subproblem (passing dual variables)
        print('Solving subproblem (AP):', card_omega, flush = True)
        
        
        routes,reduced_costs,costs = subProblemHeuristic(pi)
        
        if len(routes) == 0:
            route, reduced_cost, cost = subProblem(pi)
            print("Solved exactly")
            # Check termination condition
            if reduced_cost>=-0.005:
                print(reduced_cost)
                print("Column generation stops! \n", flush = True)
                break
            else:
                routes = [route]
                reduced_costs = [reduced_cost]
                costs = [cost]
        else:
            print("Solved heuristically")
                
        for i in range(len(routes)):
            route = routes[i]
            reduced_cost = reduced_costs[i]
            cost = costs[i]
            print('Minimal reduced cost (via CSP):', reduced_cost, '<0.', flush = True)
            print('Adding column...', flush = True)
            print('Route: ', route)
            card_omega+=1
            Omega[card_omega] = route
            a_star = obtainColumn(route)
            newCol = Column(a_star, modelMP.getConstrs())
            y[card_omega] = modelMP.addVar(vtype = GRB.CONTINUOUS, obj = cost, lb = 0, column = newCol, name = "y[%d]" %card_omega)
            # Update master model
            modelMP.update()

def columnGenerationHeuristic():
    """
    This function performs a Column Generation procedure
    """
    #print("Entering function columnGeneration.",flush = True)
    card_omega = len(y)
    import time  
    #while (time.time()-initialization_time_stamp)<=t_max:
    while True:
        print("t=", (time.time()-initialization_time_stamp)," t_max=", t_max, sep = "", end ="\n")
        print('Solving Master Problem (MP)...', flush = True)
        modelMP.optimize()
        print('Value of LP relaxation of MP: ', modelMP.getObjective().getValue(), flush = True)
        #Retrieving duals of master problem
        pi = {i:ctr.pi for i,ctr in NodeCtr.items()}

        #Solve subproblem (passing dual variables)
        print('Solving subproblem (AP):', card_omega, flush = True)
        
        routes,reduced_costs,costs = subProblemHeuristic(pi)
        
        if len(routes) == 0:
            print("Column generation stops! \n", flush = True)
            break
                
        for i in range(len(routes)):
            route = routes[i]
            reduced_cost = reduced_costs[i]
            cost = costs[i]
            print('Minimal reduced cost (via CSP):', reduced_cost, '<0.', flush = True)
            print('Adding column...', flush = True)
            print('Route: ', route)
            card_omega+=1
            Omega[card_omega] = route
            a_star = obtainColumn(route)
            newCol = Column(a_star, modelMP.getConstrs())
            y[card_omega] = modelMP.addVar(vtype = GRB.CONTINUOUS, obj = cost, lb = 0, column = newCol, name = "y[%d]" %card_omega)
            # Update master model
            modelMP.update()
 
############# Auxiliary Function ###########
def calculateRouteCostAndTime(route):
    "computes the cost and time of a route"
    cost = c_f
    time = 0
    for r in range(len(route)-1):
        cost+=c[(route[r], route[r+1])]
        time+=t_tilde[(route[r], route[r+1])]
    if time>L:
        cost+= c_o*(time-L)
    return(round(cost,2),round(time,2))


def calculateRouteReducedCost(route, pi):
    "computes the reduced cost of a route given the dual variables pi"
    (route_cost, route_time) = calculateRouteCostAndTime(route)
    reduced_cost = route_cost-sum(pi[i] for i in route if i>=1 and i<=n)
    return (round(reduced_cost,2),round(route_cost,2), round(route_time,2))


def obtainColumn(route):
    "obtains the 'column' associated with a route"
    a_star = [0]*len(N)
    for r in route:
        if r>=1 and r<=n:
            a_star[r-1] = 1
    return a_star


def calculateModifiedCost(pi):
    "computes the composite cost given a multiplier mu"
    c_mod = {}
    for (i,j) in A:
        if i>=1 and i<=n:
            c_mod[(i,j)] = round(c[(i,j)] - pi[i],2)
        else:
            c_mod[(i,j)] = round(c_f + c[(i,j)],2)
    return c_mod
    
def recoverDummySubtour(dummy_subtour):
    "recovers the path associated with a dummy subtour"
    heuristic_path = []
    begin = dummy_subtour.index(0)
    heuristic_path = dummy_subtour[begin:]
    heuristic_path+=dummy_subtour[0:begin]
    return heuristic_path


#deposits
location[0] = (0.0, 0.0)
location[n+1] = (0.0, 0.0)

for i in range(1,n+1):
    xcoord = np.random.uniform(low = -200.0, high = 200.0)
    ycoord = np.random.uniform(low = -200.0, high = 200.0)
    location[i] = (xcoord, ycoord)

#distances
d = {}
for i in N:
    for j in N:
        if i>j:
            d[(i,j)] = round(math.sqrt((location[i][0] - location[j][0])**2+(location[i][1] - location[j][1])**2), decimals)
            d[(j,i)] = d[(i,j)]
#deposits
for i in N:
    d[(0,i)] = round(math.sqrt((location[0][0] - location[i][0])**2+(location[0][1] - location[i][1])**2), decimals)
    d[(i,n+1)] = d[(0,i)]
    
#Arcos
A = list(d.keys())     

#service time
s = {}
for i in N:
    s[i] = mean_service_time    

#travel times
t = {}
mu_aux = {}
sigma_aux = {}
for i,j in A:
    m = d[(i,j)]
    m_prima = m/d[(i,j)]*1.6
    v = (d[(i,j)]/1.6)*(-0.4736 + 0.9936*m_prima)
    mu = math.log(m**2/(math.sqrt(v+m**2)))
    sigma = math.sqrt(math.log(1+v/m**2))
    t[(i,j)] = round(math.exp(mu + sigma**2/2), decimals)
    mu_aux[(i,j)] = mu
    sigma_aux[(i,j)] = sigma

t_tilde = {}
for i,j in A:
    t_tilde[(i,j)] = t[(i,j)]
    if i>=1 and i<=n:
        t_tilde[(i,j)] += s[i]
        t_tilde[(i,j)] = round(t_tilde[(i,j)],decimals)

#costs
c = {}
for (i,j) in A:
    c[(i,j)] = round(c_t*t[(i,j)], decimals)

################ INITIAL SOLUTION (MST) ################
import time
initial_time_stamp = time.time()
G = nx.Graph()

#nodes
G.add_nodes_from(N)
G.add_node(0)

#edges
#start
for j in N:
    G.add_edge(0,j, weight = c[(0,j)])

#intermediate
for i in N:
    for j in N:
        if i!=j:
            G.add_edge(i,j, weight = c[(i,j)])

#not using a vehicle
#G.add_edge(0, n+1, weight = 0)

mst = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="prim")
edgelist = list(mst)

#Preorder of the tree
edgelist
G_tree = nx.Graph()
G_tree.add_edges_from(edgelist)
G_tree.add_nodes_from(G.nodes)
route = list(nx.dfs_preorder_nodes(G_tree, source=0, depth_limit=len(N)))
route.append(n+1)    
print(route)   

######### Split (Shortest path - DAG) ################
giantTour = route[0:-1]

G_dag =nx.DiGraph()
G_dag.add_nodes_from(giantTour)

for i in range(len(giantTour)):
    for j in range(i+1, len(giantTour)):
        if i == 0:
            (route_cost, route_time) = calculateRouteCostAndTime(giantTour[i:j+1]+[n+1])
        else:
            (route_cost, route_time) = calculateRouteCostAndTime([0]+giantTour[i+1:j+1]+[n+1])
        G_dag.add_edge(giantTour[i], giantTour[j], weight = route_cost)

bf = nx.bellman_ford_path(G_dag, source = 0, target = giantTour[-1])

initial_routes = []
index = 1
until_node = bf[index]
path =[0]
for i in giantTour[1:]:
    path.append(i)
    if i == until_node:
        if index<len(bf)-1:
            index+=1
            until_node = bf[index]
        path.append(n+1)
        initial_routes.append(path.copy())
        path = [0]
print(initial_routes)        
import time
initialization_time = time.time() - initial_time_stamp
initialization_time_stamp = time.time()     

############# SET COVERING ###############
modelMP = Model("Master problem")
modelMP.Params.Presolve = 0
modelMP.Params.Cuts = 0
modelMP.Params.OutputFlag = 0
#Set covering formulation
NodeCtr = {}                #Node covering constraints
y = {}                      #Route selection variables
card_omega = 1             #Total number of routes in Omega
a = {}                     #Columns of the routes
Omega = {}                 #Routes
Z_IS = 0

#Initial set-covering model  (PETALS INITIALIZATION + Initial long route)
for i in N:
    (route_cost, route_time) = calculateRouteCostAndTime([0,i,n+1])
    y[card_omega] = modelMP.addVar(vtype = GRB.CONTINUOUS, obj = route_cost, lb = 0, name = "y[%d]" %card_omega)
    a[card_omega] = obtainColumn([0,i,n+1])
    Omega[card_omega] = [0,i,n+1]
    card_omega+=1

for route in initial_routes:
    (route_cost, route_time) = calculateRouteCostAndTime(route)
    y[card_omega] = modelMP.addVar(vtype = GRB.CONTINUOUS, obj = route_cost, lb = 0, name = "y[%d]" %card_omega)
    a[card_omega] = obtainColumn(route)
    Omega[card_omega] = route
    card_omega+=1
    Z_IS+=route_cost

#constraints    
for i in N:
    NodeCtr[i] = modelMP.addConstr(sum(a[r][i-1]*y[r] for r in range(1,card_omega))>=1, "Set_Covering_[%i]"%i) #Set covering constraints

modelMP.modelSense = GRB.MINIMIZE #Objective function
#modelMP.write("MP_0.lp")

#if t_max == 0:
#    columnGenerationHeuristic()
#else:
#    columnGeneration()
modelMP.optimize()
        
# Print relaxed solution
Z_LP = modelMP.objVal
final_routes_indices = []
is_integer = True
print('Relaxed master problem:')
print('Total cost: %g' % modelMP.objVal)
for v in modelMP.getVars():
    if v.x>0.01:
        print('%s=%g' % (v.varName, v.x))
        final_routes_indices.append(int(v.varName[2:-1]))
    if is_integer and v.x>0 and v.x<1:
        is_integer = False
        
#modelMP.write("ModelMP.lp")
#modelMP.write("ModelMP.sol")
if not is_integer:
    final_routes_indices = []
    # Solve integer program
    for v in modelMP.getVars():
        v.setAttr("Vtype", GRB.INTEGER)
    modelMP.optimize()
    # Print integer solution
    print('(Heuristic) integer master problem:')
    print('Route time: %g' % modelMP.objVal)
    for v in modelMP.getVars():
        if v.x > 0.5:
            print('%s %g' % (v.varName, v.x))
            final_routes_indices.append(int(v.varName[2:-1]))
Z_IP = modelMP.objVal

#Recover final routes (post-optimization procedure)
final_routes = []
visited = {i:False for i in N}
for i in range(len(final_routes_indices)):
    index = final_routes_indices[i]
    path = Omega[index]
    toDelete = []
    for p in path:
        if p>=1 and p<=n:
            if visited[p] == False:
                visited[p] = True
            else:
                toDelete.append(p)
    for delete in toDelete:
        path.remove(delete)
    (cost, time) = calculateRouteCostAndTime(path)
    final_routes.append((path,cost,time))
    print(path, cost, time) 


import time     
column_generation_time = time.time()-initialization_time_stamp
column_generation_time_stamp = time.time()
print("--- Column Generation time %s seconds ---" % (column_generation_time), flush = True)


############### SIMULATION ##################
w = {i:0 for i in V}
for (route,cost,time) in final_routes:
    route_arcs = [(route[i],route[i+1]) for i in range(len(route)-1)]
    REPLICATIONS = 100
    replication_info_by_node = {i:[] for i in route}
    for k in range(1,REPLICATIONS+1):
        arrival_time = 0
        replication_info_by_node[0] = arrival_time
        for arc in route_arcs:
            if np.random.binomial(1, 0.1) == 0: 
                arrival_time += np.random.lognormal(mu_aux[arc],sigma_aux[arc],size=1)[0]
                replication_info_by_node[arc[1]].append(arrival_time)
                arrival_time+=np.random.exponential(scale = mean_service_time, size = 1)[0]
    for i in route:
        if i>=1 and i<=n:
            w[i] = np.percentile(replication_info_by_node[i], alpha)

import time     
simulation_time = time.time()-column_generation_time_stamp


############### PRINT RESULTS ##################
original = sys.stdout
import time     
total_time = initialization_time + column_generation_time + simulation_time
sys.stdout = open("./Experiment3.txt", "a")
print(n, round(total_time,2), round(initialization_time,2), round(column_generation_time,2), round(simulation_time,2), round(Z_IS,2), round(Z_LP,2), round(Z_IP,2), sep =",")
sys.stdout = original

