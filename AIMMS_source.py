'''
15th MOPTA Competition

OptiCoffee Team
Universidad de los Andes

Python back-end for the AIMMS application
'''
import gurobipy as gb
import networkx as nx
import numpy as np; import pandas as pd; from time import process_time
import pickle
import os

path = os.getcwd()

def generate_simulation_KPIs():
    pass
    

def reconstruct_routes(Results):
    # Reconstructing the routes
    schedules = {} 

    active_stations = 0 
    avg_distance = 0 
    visited = [] 

    avg_charging_time = 0 

    avg_utilization = 0 
    active_chargers = 0 

    avg_waiting_time = 0 

    avg_driving_charging_cost = 0 
    latest_finalization = 0 

    
    for s in S:
        charger_num = Results['Chargers'][s]
        
        # Station has no chargers assigned
        if charger_num == 0: 
            continue

        # Retrieve information from all scenarios
        for sc in range(25):
            station_is_active = False
            K, K_s, S_k, a, t = load_pickle(path+"/Data/",sc)
            file = open(f'{path}/Data/p/p_{sc}','rb'); p = pickle.load(file); file.close()
            active_char = 0
            
            routes = Results[idx2][sc]['routes'][s]

            schedules[idx][sc][s] = list()

            # Station has no assigned vehicles that scenario (unactive)
            if len(routes) == 0: 
                continue
            
            station_is_active = True
            active_chargers[idx][sc] += Results[idx2]['Chargers'][s]

            for route in routes:
                schedule = {'vehicles':list(), 'arrive':list(),'wait':list(), 'start':list(), 'end':list()}

                for pos, vehicle in enumerate(route):
                    visited[idx][sc].append(vehicle)
                    avg_distance[idx][sc] += distances[s,vehicle]/len(Results[idx2][sc]['total_total'])
                    schedule['vehicles'].append(vehicle)

                    if pos == 0:
                        # print(sc,vehicle,s)
                        schedule['arrive'].append(a[vehicle,s])
                        schedule['start'].append(a[vehicle,s])
                        schedule['wait'].append(0)
                        schedule['end'].append(a[vehicle,s] + t[vehicle,s])
                    else:
                        schedule['arrive'].append(a[vehicle,s])

                        if schedule['end'][-1] > a[vehicle,s]:
                            schedule['wait'].append(schedule['end'][-1] - a[vehicle,s])
                        else:
                            schedule['wait'].append(0)
                        avg_waiting_time[idx][sc] += schedule['wait'][-1]/len(Results[idx2][sc]['total_total'])

                        schedule['start'].append(schedule['arrive'][-1] + schedule['wait'][-1])
                        schedule['end'].append(schedule['start'][-1] + t[vehicle,s])

                    avg_charging_time[idx][sc] += (schedule['end'][-1] - schedule['start'][-1])/len(Results[idx2][sc]['total_total'])
                    avg_utilization[idx][sc] += schedule['end'][-1] - schedule['start'][-1]

                    avg_driving_charging_cost[idx][sc] += p[vehicle,s] * 0.0388  + distances[s,vehicle] * 0.041
                
                schedules[idx][sc][s].append(schedule)

            if station_is_active: 
                active_stations[idx][sc] += 1
                

            latest_finalization[idx][sc] = max(schedule['end'][-1], 14)

    for sc in range(25):
        avg_utilization[idx][sc] = (avg_utilization[idx][sc] / active_chargers[idx][sc])

    return schedules, active_stations, avg_distance, visited, avg_charging_time, avg_utilization, avg_waiting_time, avg_driving_charging_cost, latest_finalization
