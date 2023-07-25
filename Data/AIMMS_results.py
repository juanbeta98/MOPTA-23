#%%
# Modules
import pandas as pd
import os 
import pickle
from scipy import stats

path = os.getcwd()

#%%
# Read main data
stations = pd.read_csv(path+"/fuel_stations.csv")
profile = pd.read_excel(path+"/CountyProfile.xlsx", index_col=0)

northern = (-79.761960, 42.269385)
southern = (-76.9909,39.7198)
western = (-80.519400, 40.639400)
eastern = (-74.689603, 41.363559)


latitudes = list()
longitudes = list()
profiles = list()
cont = 0

for i in stations.index:
    long = (stations["Longitude"][i]-western[0])*52-1
    lat  = (stations["Latitude"][i]-southern[1])*69-7

    if (long <= 290) & (lat <= 150) & (lat >= 0) & (long >= 0) and i+1:
        latitudes.append(stations['Latitude'][i])
        longitudes.append(stations['Longitude'][i])
        try:
            profiles.append(profile["Type"][stations["ZIP"][i]])
        except:
            profiles.append('Urban')
            cont += 1

#%%
# Load chosen stations and number of chargers (greedy)
file = open('../Results/Optimal/S', 'rb');open_stations = pickle.load(file);file.close()
file = open('../Results/Optimal/n', 'rb');number_of_chargers = pickle.load(file);file.close()


lat = list();lon = list();g_char = list();prof = list()
for i in range(len(longitudes)):
    if i+1 in open_stations:
        lat.append(latitudes[i]); lon.append(longitudes[i])
        g_char.append(number_of_chargers[i+1])
        prof.append(profiles[i])

ddd = {'latitude':lat, 'longitude':lon, 'Greedy Chargers':g_char,'profile':prof}

ddff = pd.DataFrame(data = ddd)
ddff.to_excel('AIMMS_results.xlsx')

#%% 
# Recostructing routes function
os.chdir('../')
from source import *
os.chdir(path)
         
def reconstruct_routes(S,distances,Results,sim_num:int=25):
    simulations = range(sim_num)

    # Reconstructing the routes
    schedules = {sc:{} for sc in simulations}

    active_stations = {sc:0 for sc in simulations}
    avg_distance = {sc:0 for sc in simulations}
    visited = {sc:[] for sc in simulations}

    avg_charging_time = {sc:0 for sc in simulations}

    avg_utilization = {sc:0 for sc in simulations}
    active_chargers = {sc:0 for sc in simulations}

    avg_waiting_time = {sc:0 for sc in simulations}

    avg_driving_charging_cost = {sc:0 for sc in simulations}
    latest_finalization = {sc:0 for sc in simulations}
            
    print('Starting charging routes reconstruction:')

    for idx,s in enumerate(S):
        if idx%10==0:print(f'{round(100*idx/len(S),2)}%')
        charger_num = Results[f'Chargers'][s]
        return_K = dict()

        # Retrieve information from all scenarios
        for sc in simulations:
            station_is_active = False
            K, K_s, S_k, a, t = load_pickle(path,sc)
            return_K[sc] = K
            file = open(f'{path}/p/p_{sc}','rb'); p = pickle.load(file); file.close()
            active_char = 0
            
            routes = Results[sc]['routes'][s]

            schedules[sc][s] = list()

            # Station has no assigned vehicles that scenario (unactive)
            if len(routes) == 0: 
                continue
            
            station_is_active = True
            active_chargers[sc] += Results['Chargers'][s]

            for route in routes:
                schedule = {'vehicles':list(), 'arrive':list(),'wait':list(), 'start':list(), 'end':list()}

                for pos, vehicle in enumerate(route):
                    visited[sc].append(vehicle)
                    avg_distance[sc] += distances[s,vehicle]/len(Results[sc]['total_total'])
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
                        avg_waiting_time[sc] += schedule['wait'][-1]/len(Results[sc]['total_total'])

                        schedule['start'].append(schedule['arrive'][-1] + schedule['wait'][-1])
                        schedule['end'].append(schedule['start'][-1] + t[vehicle,s])

                    avg_charging_time[sc] += (schedule['end'][-1] - schedule['start'][-1])/len(Results[sc]['total_total'])
                    avg_utilization[sc] += schedule['end'][-1] - schedule['start'][-1]

                    avg_driving_charging_cost[sc] += p[vehicle,s] * 0.0388  + distances[s,vehicle] * 0.041
                
                schedules[sc][s].append(schedule)

            if station_is_active: 
                active_stations[sc] += 1
                

            latest_finalization[sc] = max(schedule['end'][-1], 14)

    for sc in range(25):
        avg_utilization[sc] = (avg_utilization[sc] / active_chargers[sc])

    return schedules, active_stations, avg_distance, visited, avg_charging_time, avg_utilization, \
                avg_waiting_time, avg_driving_charging_cost, latest_finalization, return_K

def weight_scenarios(iterable):
    return round(sum(iterable[sc] for sc in range(25))/25,2)


def compute_interval(iterable):
    # Compute mean and standard deviation
    mean = np.mean(iterable)
    std_dev = np.std(iterable)

    # Degrees of freedom
    df = len(iterable) - 1

    # Alpha level (e.g., 0.05 for 95% confidence interval)
    alpha = 0.025

    # Calculate t-value
    t_value = stats.t.ppf(1 - alpha/2, df)

    # Calculate standard error
    std_err = std_dev / np.sqrt(len(iterable))

    # Calculate confidence interval
    lower_bound = mean - t_value * std_err
    upper_bound = mean + t_value * std_err

    return round(lower_bound,2), round(upper_bound,2)



#%%
# Recostructing routes
file = open('../Results/Optimal/distances', 'rb');distances = pickle.load(file);file.close()
file = open(f'../Results/Optimal/results', 'rb')
Results = pickle.load(file)
file.close()

schedules, active_stations, avg_distance, visited, avg_charging_time, avg_utilization, \
avg_waiting_time, avg_driving_charging_cost, latest_finalization, K = reconstruct_routes(open_stations,distances,Results,25)

#%% 
# Print overall performance
print(f'- Number of active stations: {sum([1 for value in Results["Chargers"].values() if value != 0])}')
print(f'- Number of chargers: {sum(Results["Chargers"].values())} \n')

greedy_number_of_chargers = sum(Results['Chargers'].values())
greedy_number_of_stations = sum(1 for i,j in Results['Chargers'].items() if j > 0)
SC = range(25)

print('\n \n--------------------------- Greedy -----------------------------')
print(f'\t \t \tavg \tmin \tlow \thigh \tmax')
print(f'Opended stations \t{greedy_number_of_stations} \t- \t- \t- \t-')
print(f'Chargers \t \t{greedy_number_of_chargers} \t- \t- \t- \t-')
# print(f'Avg stress index  \t{round(sum(j for i,j in original_stress.items() if i in Results[1][0]["routes"].keys())/greedy_number_of_stations,2)} \t- \t- \t- \t-')

val = active_stations; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg active stations \t{weight_scenarios(val)} \t{min(val.values())} \t{lower_bound} \t{upper_bound} \t{max(val.values())}')

val = [len(K[sc])/greedy_number_of_stations for sc in SC]; lower_bound, upper_bound = compute_interval(val)
print(f'Avg serviced car \t{round(sum(val)/25,2)} \t{round(min(val),2)} \t{lower_bound} \t{upper_bound} \t{round(max(val),2)}')

val = avg_charging_time; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg charging time \t{weight_scenarios(val)} \t{round(min(val.values()),2)} \t{lower_bound} \t{upper_bound}\t{round(max(val.values()),2)}')

val = avg_distance; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg traveled distance \t{weight_scenarios(val)} \t{round(min(val.values()),2)}  \t{lower_bound} \t{upper_bound} \t{round(max(val.values()),2)}')

val = [avg_utilization[i]/latest_finalization[i] for i in SC]; lower_bound, upper_bound = compute_interval(val)
print(f'Avg Utilization: \t{round(sum(val)/25,2)} \t{round(min(val),2)}  \t{lower_bound} \t{upper_bound} \t{round(max(val),2)}')

val = avg_waiting_time; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg waiting time \t{weight_scenarios(val)}\t{round(min(val.values()),2)}  \t{lower_bound} \t{upper_bound} \t{round(max(val.values()),2)}')

val = avg_driving_charging_cost; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Expected cost \t\t{round(weight_scenarios(val),1)} {round(min(val.values()),1)}  {round(lower_bound,1)} {round(upper_bound,1)} {round(max(val.values()),1)}')
# %%
