# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:39:05 2021
MOPTA - Instance Generator
@author:The optimistics
"""
#packages
from gurobipy import *
import numpy as np
import math
import time
import sys
import networkx as nx
import folium
import random
import scipy.stats
import dataio

#colors (map in AIMMS)
COLORS={0:"maroon", 1:"green", 2:"orange", 3:"navy", 4:"purple", 5:"violet", 6:"brown", 7:"gray", 8:"blue",9: "black", 10:"coral", 11:"lime", 12:"yellow", 13:"pink", 14:"indigo", 15:"red",16:"aqua", 17:"teal", 18:"black"}

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


# Pricing problem (elementary shortest path)
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
    sp.update()
    sp.optimize()
    
    arcs_aux = {(i,j):x[i,j].x for (i,j) in A if x[(i,j)].x>0.1}
    arcs = {i:j for (i,j) in A if x[(i,j)].x>0.1}
    (subtours, dummy_subtour) = findSubtours(arcs)
    elementary_dummy_path = recoverDummySubtour(dummy_subtour)
    (rc_dummy, cost_dummy, time_dummy) = calculateRouteReducedCost(elementary_dummy_path, pi)
    import time
    while len(subtours)>0 and rc_dummy>=-0.05 and (time.time()-initialization_time_stamp)<=t_max:
        for tour in subtours:
            for k in tour:
                sp.addConstr(sum(x[i,j] for i in tour for j in tour if i!=j)<=sum(x[i,j] for i in tour for j in N+[n+1] if i!=k and j!=i))
            
        sp.optimize()
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
     
# Subproblem Heuristic
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
    while (time.time()-initialization_time_stamp)<=t_max:
        print("t=", (time.time()-initialization_time_stamp),", t_max=", t_max, sep = "", end ="\n")
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

#Column generarion heuristic
def columnGenerationHeuristic():
    """
    This function performs a Column Generation procedure
    """
    #print("Entering function columnGeneration.",flush = True)
    card_omega = len(y)
    import time  
    while (time.time()-initialization_time_stamp)<=t_max:
        print("t=", (time.time()-initialization_time_stamp),", tmax=",t_max, sep = "", end ="\n")
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

#Defines the the appointment time in time format
def hourFormat(s, n):
      h, m = map(int, s[:].split(':'))
      h %= 24
      t = h * 60 + m + n
      h, m = divmod(t, 60)
      h %= 24
      h = int(h)
      m = int(m)
      return str("{:02d}:{:02d}".format(h, m))

#runs the H-SARA problem and return the information for AIMMS
def my_model(input:dict):
    
    global n, L, c_f, c_t, c_o, alpha, c, t, N, A, V, y, modelMP, NodeCtr, card_omega, Omega, t_tilde,initialization_time_stamp, t_max
    
    #recover information from AIMMS interface
    L, c_f, c_t, c_o, alpha, mean_service_time, random_instance,n, lat, lon, prob, hour, minute, t_max, sol_method = dataio.dataFromDict(input)
    location = {} 
    new_lat = {}
    new_lon = {}
    new_prob = {}
        
    print(prob)
    print(n)
    
    if len(prob)<n:
        prob = prob + [0.1]*(n-len(prob))
    elif len(prob)>n:
        prob = prob[:n]
        
    #randon instance generator
    if random_instance == 1:
        
        N = list(range(1,n+1))
        new_prob = {i:0.1 for i in N}
        new_prob[0]=0
        
        N = list(range(1,n+1))          #set o nodes
        for i in range(1,n+1):
            coordx = np.random.uniform(low = -25.0, high = 25.0)
            coordy = np.random.uniform(low = -25.0, high = 25.0)
            location[i] = (coordx, coordy)
            new_lat[i] = coordx/110.574+40.36
            new_lon[i] = coordy/(111.320*math.cos(new_lat[i]))-75.38
         
        location[0] = (0.0, 0.0)
        location[n+1] = location[0]
        new_lat[0] = 40.36
        new_lon[0] = -75.38
        
        solution_to_return = {(i,j):0 for i in N+[0] for j in N+[0]} 

        # print solution in output.txt
        original = sys.stdout
        sys.stdout = open("./output.txt", "w")
        print("Arcs(l,u):=data{};\n")
        print("XCoordinate(l):=data")
        print(new_lat, end = ";\n")
        print("YCoordinate(l):=data")
        print(new_lon, end = ";\n")
        print("CancellationProbability(l):=data")
        print(new_prob, end = ";\n")
        routeThatVisits = {i:0 for i in range(0,n+1)}
        print("RouteOfNode(l):=data")
        print(routeThatVisits)
        print(";", end = "\n")
        w = {i:0 for i in N+[0]}
        print("AppointmentTime(l):=data{};\n")
        #print(w, end = ";\n")
        print("TotalCost:=", 0, sep = "", end = "; \n")
        print("ExpectedOvertimeCost:=", 0, sep = "", end = "; \n")
        print("ExpectedTravelTimeCost:=", 0, sep = "", end = "; \n")
        print("HiringCost:=", 0, sep ="", end=";\n")
        print("NumberOfRoutes:=",0, sep = "", end = "; \n")
        print("RouteNumber := data{}; \n")
        print("Routes(r):=data{}; \n")
        print("RouteOfNode(l):=data{}; \n")
        print("RouteColor(r):=data{}; \n")
        print("RouteCost(r):=data{}; \n")
        print("RouteTime(r):=data{}; \n")
        print("RouteSequence(r):=data{}; \n")
        size ={i:8 for i in range(1,n+1)}
        size[0] = 15
        print("NodeSize(l):=data")
        print(size, end = ";\n")
        icons ={i:"aimms-store" for i in range(1,n+1)}
        icons[0] = "aimms-office"
        print("FactoriesIcons(l):=data{")
        for i in range(0, n+1):
            if i < n:
                print(i, ":", '"'+icons[i]+'"', sep= "", end = ",")
            else:
                print(i, ":", '"'+icons[i]+'"', sep= "", end = "}; \n")
        sys.stdout = original
    
    elif random_instance == 2:
        
        N = list(range(1,n+1))
        new_lat = {i:lat[i] for i in [0]+N}
        new_lon = {i:lon[i] for i in [0]+N}
        new_prob = {i:prob[i-1] for i in N}
        new_prob[0]=0
        
        # print solution in output.txt
        original = sys.stdout
        sys.stdout = open("./output.txt", "w")
        print("Arcs(l,u):=data{};\n")
        print("XCoordinate(l):=data")
        print(new_lat, end = ";\n")
        print("YCoordinate(l):=data")
        print(new_lon, end = ";\n")
        print("CancellationProbability(l):=data")
        print(new_prob, end = ";\n")
        routeThatVisits = {i:0 for i in range(0,n+1)}
        print("RouteOfNode(l):=data")
        print(routeThatVisits)
        print(";", end = "\n")
        w = {i:0 for i in N+[0]}
        print("AppointmentTime(l):=data{};\n")
        #print(w, end = ";\n")
        print("TotalCost:=", 0, sep = "", end = "; \n")
        print("ExpectedOvertimeCost:=", 0, sep = "", end = "; \n")
        print("ExpectedTravelTimeCost:=", 0, sep = "", end = "; \n")
        print("HiringCost:=", 0, sep ="", end=";\n")
        print("NumberOfRoutes:=",0, sep = "", end = "; \n")
        print("RouteNumber := data{}; \n")
        print("Routes(r):=data{}; \n")
        print("RouteOfNode(l):=data{}; \n")
        print("RouteColor(r):=data{}; \n")
        print("RouteCost(r):=data{}; \n")
        print("RouteTime(r):=data{}; \n")
        print("RouteSequence(r):=data{}; \n")
        size ={i:8 for i in range(1,n+1)}
        size[0] = 15
        print("NodeSize(l):=data")
        print(size, end = ";\n")
        icons ={i:"aimms-store" for i in range(1,n+1)}
        icons[0] = "aimms-office"
        print("FactoriesIcons(l):=data{")
        for i in range(0, n+1):
            if i < n:
                print(i, ":", '"'+icons[i]+'"', sep= "", end = ",")
            else:
                print(i, ":", '"'+icons[i]+'"', sep= "", end = "}; \n")
        sys.stdout = original
        
    else:
        #Solve optimization problem
        N = list(range(1,n+1))          #set o nodes
        V = list(range(0,n+2))          #set of vertices
        
        for i in [0]+N:
            coordx = (lat[i] - 40.36)*110.574
            coordy = (lon[i] + 75.38)*(111.320*math.cos(lat[i]))
            new_lat[i] = lat[i]
            new_lon[i] = lon[i]
            location[i] = (coordx,coordy)
            
        #deposits
        location[n+1] = location[0]
        
        #distances
        d = {}
        for i in N:
            for j in N:
                if i>j:
                    d[(i,j)] = round(math.sqrt((location[i][0] - location[j][0])**2+(location[i][1] - location[j][1])**2), 2)
                    d[(j,i)] = d[(i,j)]
        #deposits
        for i in N:
            d[(0,i)] = round(math.sqrt((location[0][0] - location[i][0])**2+(location[0][1] - location[i][1])**2), 2)
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
            t[(i,j)] = round(math.exp(mu + sigma**2/2), 2)
            mu_aux[(i,j)] = mu
            sigma_aux[(i,j)] = sigma
        
        t_tilde = {}
        for i,j in A:
            t_tilde[(i,j)] = t[(i,j)]
            if i>=1 and i<=n:
                t_tilde[(i,j)] += s[i]
                t_tilde[(i,j)] = round(t_tilde[(i,j)],2)
            
        #costs
        c = {}
        for (i,j) in A:
            c[(i,j)] = round(c_t*t[(i,j)], 2)
    
        ################ INITIAL SOLUTION (MST) ################
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
        
        mst = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="prim")
        edgelist = list(mst)
        
        ####### SOLUTION #########
                  
        #Preorder of the tree
        edgelist
        G_tree = nx.Graph()
        G_tree.add_edges_from(edgelist)
        G_tree.add_nodes_from(G.nodes)
        route = list(nx.dfs_preorder_nodes(G_tree, source=0, depth_limit=len(N)))
        route.append(n+1)    
        
        
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
        
        import time
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
        
        #constraints    
        for i in N:
            NodeCtr[i] = modelMP.addConstr(sum(a[r][i-1]*y[r] for r in range(1,card_omega))>=1, "Set_Covering_[%i]"%i) #Set covering constraints
        
        modelMP.modelSense = GRB.MINIMIZE #Objective function
                      
        # Column generation loop
        if sol_method == 'Exact Method':
            columnGeneration()
        else:
            columnGenerationHeuristic()
            
        # Print relaxed solution
        final_routes_indices = []
        is_integer = True
        modelMP.optimize()
        
        print('Relaxed master problem:')
        print('Total cost: %g' % modelMP.objVal)
        for v in modelMP.getVars():
            if v.x>0.01:
                print('%s=%g' % (v.varName, v.x))
                final_routes_indices.append(int(v.varName[2:-1]))
            if is_integer and v.x>0 and v.x<1:
                is_integer = False
        
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
        
        #Recover final routes (post-optimization procedure)
        arcs_used = {}
        final_routes = []
        visited = {i:False for i in N}
        routeThatVisits = {i:-1 for i in N}
        overtimeCost = 0
        for i in range(len(final_routes_indices)):
            index = final_routes_indices[i]
            path = Omega[index]
            toDelete = []
            for j in range(len(path)):
                p = path[j]
                if j<len(path)-1:
                    arcs_used[(p,path[j+1])] = 1
                if p>=1 and p<=n:
                    if visited[p] == False:
                        visited[p] = True
                        routeThatVisits[p]= i+1
                    else:
                        toDelete.append(p)
            for delete in toDelete:
                path.remove(delete)
            (cost, time) = calculateRouteCostAndTime(path)
            if time>L:
                overtimeCost+=(time-L)*c_o
            final_routes.append((path,cost,time))
            print(path, cost, time) 
        
        solution_to_return = {}
        
        for (i,j) in A:
            if j == n+1:
                solution_to_return[(i,0)] = arcs_used.get((i,j),0)
            elif i != n+1:
                solution_to_return[(i,j)] = arcs_used.get((i,j),0)
                
        ############### SIMULATION ##################
        w = {i:str(0) for i in ([0] + N)}
        for (route,cost,time) in final_routes:
            route_arcs = [(route[i],route[i+1]) for i in range(len(route)-1)]
            REPLICATIONS = 100
            replication_info_by_node = {i:[] for i in route}
            for k in range(1,REPLICATIONS+1):
                arrival_time = 0
                replication_info_by_node[0] = arrival_time
                for arc in route_arcs:
                    if arc[1]<=n and np.random.binomial(1, prob[arc[1]-1]) == 0:
                        arrival_time += np.random.lognormal(mu_aux[arc],sigma_aux[arc],size=1)[0]
                        replication_info_by_node[arc[1]].append(arrival_time)
                        arrival_time+=np.random.exponential(scale = mean_service_time, size = 1)[0]
            for i in route:
                if i>=1 and i<=n:
                    perc = round(np.percentile(replication_info_by_node[i], alpha),2)
                    w[i] = str(hourFormat(hour+":"+minute, int(round(perc,0))))
        w[0] = hour+":"+minute
            
        ColorsNeeded = {}
        for key,val in COLORS.items():
            if key<len(final_routes):
                ColorsNeeded[key] = '"' + val + '"'
        
        # print solution in output.txt
        original = sys.stdout
        sys.stdout = open("./output.txt", "w")
        print("Arcs(l,u):=data")
        print(solution_to_return)
        print(";", end = "\n")
        print("TotalCost:=", round(modelMP.objVal,2), sep = "", end = "; \n")
        print("ExpectedOvertimeCost:=", round(overtimeCost,2), sep = "", end = "; \n")
        print("ExpectedTravelTimeCost:=", round(modelMP.objVal-overtimeCost-len(final_routes)*c_f,2), sep = "", end = "; \n")
        print("HiringCost:=", round(len(final_routes)*c_f,2), sep ="", end=";\n")
        print("NumberOfRoutes:=",len(final_routes_indices), sep = "", end = "; \n")
        
        
        
        print("AppointmentTime(l):=data{") 

        for i in ([0] + N):
            if i<len(N):
                print(i,":", end = '"' + w[i] + '"' + ",")
            else:
                print(i,":", end = '"' + w[i] + '"'+ "};\n")
        
        
        print("XCoordinate(l):=data")
        print(new_lat, end = ";\n")
        print("YCoordinate(l):=data")
        print(new_lon, end = ";\n")
        print("RouteNumber := data{")
        for r in range(0, len(final_routes)):
            if r<len(final_routes)-1:
                print(r+1, end = ",")
            else:
                print(r+1, end = "};\n")
        print("Routes(r):=data{")
        for r in range(0, len(final_routes)):
            path = final_routes[r][0]
            print(r+1,":", end = "{")
            for j in range(0, len(path)-1):
                if j<len(path)-2:
                    print(path[j], end = ",")
                else:
                    if r<len(final_routes)-1:
                        print(path[j], end = "},")
                    else:
                        print(path[j], end = "} };\n")
        
        print("RouteOfNode(l):=data")
        print(routeThatVisits)
        print(";", end = "\n")
        print("Route(l):=data{")
        for l in range(len(N)):
            if l < len(N)-1:
                print(l+1, ":", '"'+str(routeThatVisits[l+1])+'"', sep= "", end = ",")
            else:
                print(l+1, ":", '"'+str(routeThatVisits[l+1])+'"', sep= "", end = "}; \n")
        print("RouteColor(r):=data{")
        for r in range(0, len(final_routes)):
            if r < len(final_routes)-1:
                print(r+1, ":", '"'+COLORS[r]+'"', sep= "", end = ",")
            else:
                print(r+1, ":", '"'+COLORS[r]+'"', sep= "", end = "}; \n")
        print("RouteCost(r):=data{")
        for r in range(0, len(final_routes)):
            if r < len(final_routes)-1:
                print(r+1, ":", final_routes[r][1], sep= "", end = ",")
            else:
                print(r+1, ":", final_routes[r][1], sep= "", end = "}; \n")
        print("RouteTime(r):=data{")
        for r in range(0, len(final_routes)):
            if r < len(final_routes)-1:
                print(r+1, ":", final_routes[r][2], sep= "", end = ",")
            else:
                print(r+1, ":", final_routes[r][2], sep= "", end = "}; \n")
        print("RouteSequence(r):=data{")
        for r in range(0, len(final_routes)):
            path = final_routes[r][0]
            print(r+1,":", end = '"' + "[")
            for j in range(0, len(path)-1):
                if j<len(path)-2:
                    print(path[j], end = ",")
                else:
                    if r<len(final_routes)-1:
                        print(path[j], end = "]" + '"' + ",")
                    else:
                        print(path[j], end = "]" + '"'+ "};\n")
        size ={i:8 for i in range(1,n+1)}
        size[0] = 15
        print("NodeSize(l):=data")
        print(size, end = ";\n")
        icons ={i:"aimms-store" for i in range(1,n+1)}
        icons[0] = "aimms-office"
        print("FactoriesIcons(l):=data{")
        for i in range(0, n+1):
            if i < n:
                print(i, ":", '"'+icons[i]+'"', sep= "", end = ",")
            else:
                print(i, ":", '"'+icons[i]+'"', sep= "", end = "}; \n")
        sys.stdout = original
    return None
