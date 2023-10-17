#%%
# Modules
import pandas as pd
import os 
import pickle
from scipy import stats

path = os.getcwd()


#%%

#%% Read main data
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


#%% Stres Index
# Stres Index
file = open('../MultiObjective/Configurations/Stress Index/stress_index_0.0', 'rb')
data = pickle.load(file)
file.close()

stress_index = {i:sum(data[sc][i] for sc in range(25))/25 for i in data[0].keys()}

#%% Load chosen stations and number of chargers (greedy)
# Load chosen stations and number of chargers (greedy)
file = open('../Results/Optimal/S', 'rb');open_stations = pickle.load(file);file.close()
file = open('../Results/Optimal/n', 'rb');number_of_chargers = pickle.load(file);file.close()

lat = list();lon = list();stress = list();g_char = list();prof = list()

cont = 0
for i in range(len(longitudes)):
    if i+1 in open_stations:
        lat.append(latitudes[i]); lon.append(longitudes[i])
        g_char.append(number_of_chargers[i+1])
        prof.append(profiles[i])

        stress.append(stress_index[i+1])

ddd = {'latitude':lat, 'longitude':lon,'Greedy Chargers':g_char,'stress':stress,'profile':prof}

ddff = pd.DataFrame(data = ddd)
ddff.to_excel('AIMMS_results.xlsx')

#%% Recostructing routes function
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

    exp_waiting_time = {s:0 for s in S}
    exp_utilization = {s:0 for s in S}
            
    print('Starting charging routes reconstruction:')

    for idx,s in enumerate(S):
        if idx%25==0:print(f'{round(100*idx/len(S),2)}%')
        return_K = dict()

        # Retrieve information from all scenarios
        for sc in simulations:
            station_is_active = False
            K, K_s, S_k, a, t = load_pickle(path,sc)
            return_K[sc] = K
            file = open(f'{path}/p/p_{sc}','rb'); p = pickle.load(file); file.close()
            
            routes = Results[sc]['routes'][s]
            schedules[sc][s] = list()

            # Station has no assigned vehicles that scenario (unactive)
            if len(routes) == 0:    continue
            
            station_is_active = True
            active_chargers[sc] += Results['Chargers'][s]

            number_of_vehicles = 0
            total_waiting_time = 0

            for route in routes:
                schedule = {'vehicles':list(), 'arrive':list(),'wait':list(), 'start':list(), 'end':list()}

                for pos, vehicle in enumerate(route):
                    number_of_vehicles += 1
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

                        exp_utilization[s] += schedule['end'][-1] - schedule['start'][-1]
                        total_waiting_time += schedule['wait'][-1]

                    avg_charging_time[sc] += (schedule['end'][-1] - schedule['start'][-1])/len(Results[sc]['total_total'])
                    avg_utilization[sc] += schedule['end'][-1] - schedule['start'][-1]

                    avg_driving_charging_cost[sc] += p[vehicle,s] * 0.0388  + distances[s,vehicle] * 0.041
                
                schedules[sc][s].append(schedule)

            exp_waiting_time[s] += (total_waiting_time / number_of_vehicles)/25

            if station_is_active: 
                active_stations[sc] += 1
                
            latest_finalization[sc] = max(schedule['end'][-1], 14)

    total_number_of_chargers = sum(Results['Chargers'][s] for s in S)
    for sc in range(25):
        avg_utilization[sc] = avg_utilization[sc] / (active_chargers[sc]*14*60)
    
    for s in S:
        exp_utilization[s] /= (Results['Chargers'][s] * 25 * 14 * 60)
        exp_utilization *= 100

    return schedules, active_stations, avg_distance, visited, avg_charging_time, avg_utilization, \
                avg_waiting_time, avg_driving_charging_cost, latest_finalization, return_K, exp_utilization, exp_waiting_time

def weight_scenarios(iterable):
    return round(sum(iterable[sc] for sc in range(25))/25,5)

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

    return round(lower_bound,5), round(upper_bound,5)



#%% Recostructing routes
# Recostructing routes
file = open('../Results/Optimal/distances', 'rb');distances = pickle.load(file);file.close()
file = open(f'../Results/Optimal/results', 'rb')
Results = pickle.load(file)
file.close()

schedules, active_stations, avg_distance, visited, avg_charging_time, avg_utilization, avg_waiting_time, avg_driving_charging_cost, \
    latest_finalization, K, exp_utilization, exp_waiting_time = reconstruct_routes(open_stations,distances,Results,25)

#%% Print overall performance
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
print(f'Avg serviced car \t{weight_scenarios(val)} \t{round(min(val),2)} \t{lower_bound} \t{upper_bound} \t{round(max(val),2)}')

val = avg_charging_time; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg charging time \t{weight_scenarios(val)} \t{round(min(val.values()),2)} \t{lower_bound} \t{upper_bound}\t{round(max(val.values()),2)}')

val = avg_distance; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg traveled distance \t{weight_scenarios(val)} \t{round(min(val.values()),2)}  \t{lower_bound} \t{upper_bound} \t{round(max(val.values()),2)}')

val = [avg_utilization[i]/latest_finalization[i] for i in SC]; lower_bound, upper_bound = compute_interval(val)
print(f'Avg Utilization: \t{weight_scenarios(val)} \t{round(min(val),5)}  \t{lower_bound} \t{upper_bound} \t{round(max(val),5)}')

val = avg_waiting_time; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Avg waiting time \t{weight_scenarios(val)}\t{round(min(val.values()),2)}  \t{lower_bound} \t{upper_bound} \t{round(max(val.values()),2)}')

val = avg_driving_charging_cost; lower_bound, upper_bound = compute_interval(list(val.values()))
print(f'Expected cost \t\t{round(weight_scenarios(val),1)} {round(min(val.values()),1)}  {round(lower_bound,1)} {round(upper_bound,1)} {round(max(val.values()),1)}')


#%%
# Find My Station





























lat = list();lon = list();g_char = list();prof = list()
for i in range(len(longitudes)):
    if i+1 in open_stations:
        lat.append(latitudes[i]); lon.append(longitudes[i])
        g_char.append(number_of_chargers[i+1])
        prof.append(profiles[i])

stations = {'station':list(range(len(lat))),'latitude':lat, 'longitude':lon, 'chargers':g_char, 'profile':prof}



import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface given their latitude and longitude
    in decimal degrees.
    """
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

def find_closest_stations(stations, lat, lon, max_dist):
    """
    Find the ten closest stations to the given latitude and longitude,
    within the specified maximum distance, and generate an Excel file
    with the feasible stations in a sheet called 'Results'.
    
    Parameters:
        stations (dict): The dictionary containing station information.
        lat (float): Latitude of the new coordinate.
        lon (float): Longitude of the new coordinate.
        max_dist (float): Maximum distance in miles.
        file_path (str): Path to the output Excel file.
    """
    distances = []

    for i in range(len(stations['station'])):
        station_lat = stations['latitude'][i]
        station_lon = stations['longitude'][i]
        distance_miles = haversine(lat, lon, station_lat, station_lon)

        if distance_miles <= max_dist:
            distances.append((stations['station'][i], distance_miles))

    # Sort stations based on distance (closest first)
    distances.sort(key=lambda x: x[1])

    # Get the feasible stations (maximum ten or stations within max_dist)
    feasible_stations = [station[0] for station in distances[:10]]

    # Create a DataFrame for the feasible stations
    df = pd.DataFrame({'Feasible Stations': feasible_stations})

    # Create an Excel writer and write the DataFrame to the file
    writer = pd.ExcelWriter('../FindMyStation.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Results', index=False)

    # Close the Excel writer and save the file
    writer.save()


new_lat = 40.632138
new_lon = -80.054285
max_distance = 80  # Maximum distance in miles

closest_stations = find_closest_stations(stations, new_lat, new_lon, max_distance)
print(closest_stations)


#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import PathPatch
from matplotlib.patheffects import withStroke


def generate_text_image(name, kpi, expected_performance):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Set the background color to transparent
    fig.patch.set_alpha(0)

    # Add text for the name of the plot (larger, black, bold)
    ax.text(0.5, 0.85, name, fontsize=35, color='black', weight='bold',
            ha='center', va='center', transform=ax.transAxes)

    # Add text for the KPI number
    ax.text(0.5, 0.5, f'{kpi:.2f}', fontsize=50, color='black', weight='bold',
            ha='center', va='center', transform=ax.transAxes)

    # Calculate the difference in percentage
    difference_percentage = ((kpi - expected_performance) / expected_performance) * 100
    difference_text = f'Difference: {difference_percentage:.2f}%'

    # Determine text color based on the sign of the difference
    text_color = 'green' if difference_percentage >= 0 else 'red'

    # Create path effects to add a black contour to the text
    stroke_effect = withStroke(linewidth=1, foreground='black')

    # Add text for the difference in percentage with red fill, black contour, and custom font
    ax.text(0.5, 0.2, difference_text, fontsize=35, color=text_color,
            weight='bold', ha='center', va='center', transform=ax.transAxes,
            path_effects=[stroke_effect], fontname='Arial Narrow')

    # Add horizontal line segments (wider lines)
    ax.hlines(y=0, xmin=-50, xmax=-25, color='black', linewidth=5)
    ax.hlines(y=0, xmin=25, xmax=50, color='black', linewidth=5)

    # Remove axes for a cleaner image
    ax.axis('off')

    # Save the image to the specified output file
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)

    plt.show()

generate_text_image('Service Level', 16, 100)
# %%
import numpy as np
import matplotlib.pyplot as plt


def generate_gantt_diagram_for_station(station_id, station_schedule):
    fig, ax = plt.subplots(figsize=(16, 6))

    charger_count = len(station_schedule)

    max_duration = 0
    for charger in range(charger_count):
        if 'start' in station_schedule[charger]:
            start_times = station_schedule[charger]['start']
            end_times = station_schedule[charger]['end']
            
            for start_time, end_time in zip(start_times, end_times):
                duration = end_time - start_time
                if duration > max_duration:
                    max_duration = duration
    
    for charger in range(charger_count):
        charger_schedule = []  # List to store tuples (start, duration) for each vehicle

        if 'start' in station_schedule[charger]:
            start_times = station_schedule[charger]['start']
            end_times = station_schedule[charger]['end']
            
            for start_time, end_time in zip(start_times, end_times):
                duration = end_time - start_time
                start_time *= 60
                start_time += 7*60
                duration *= 60
                charger_schedule.append((start_time, duration))

        # Plotting Gantt bars for each charger
        for idx, (start, duration) in enumerate(charger_schedule):
            color = plt.cm.tab20(idx)  # Get a unique color from the palette for each vehicle
            
            ax.barh(charger, duration, left=start, height=0.6,
                    color=color, alpha=0.6, edgecolor='black', linewidth=0.5)

    # Configure axes and labels
    ax.set_xlabel('Time',fontsize=20,weight='bold')
    ax.set_xlim(7*60, 21*60)  # X-axis limits from 7:00 to 21:00 (in minutes)
    ax.set_xticks(range(7*60, 22*60, 60))  # Every hour from 7:00 to 21:00 (in minutes)
    ax.set_xticklabels([f'{hour:02d}:00' for hour in range(7, 22)],fontsize=18)
    ax.set_yticks(range(charger_count))
    ax.set_yticklabels([f'Charger {i+1}' for i in range(charger_count)],fontsize=18,weight='bold')
    ax.set_title(f'Charging Schedule of station {station_id}',fontsize=25,weight='bold')

    plt.tight_layout()
    plt.show()


station = 3
generate_gantt_diagram_for_station(station,schedules[0][open_stations[station]])

# %%
import matplotlib.pyplot as plt
station = 1
schedules_1 = schedules[0]

def display_demand(schedules):
    charging_counts = []  # List to store the number of charging cars at each interval
    time_intervals = [i for i in range(7*60, 21*60)]

    for t in time_intervals:
        charging_cars = 0

        for station_id, station_schedule in schedules.items():
            charger_count = len(station_schedule)

            for charger in range(charger_count):
                if 'start' in station_schedule[charger]:
                    start_times = station_schedule[charger]['start']
                    end_times = station_schedule[charger]['end']
                    
                    for start_time, end_time in zip(start_times, end_times):
                        if start_time*60 <= t < end_time*60:
                            charging_cars += 1
        
        charging_counts.append(charging_cars)
    
    fig,ax = plt.subplots(figsize=(16, 6))
    plt.plot(time_intervals, charging_counts, linestyle='-', color='purple',linewidth=2.9)
    
    plt.xlabel('Time',fontsize=20,weight='bold')
    plt.ylabel('Number of Charging Cars',fontsize=18,weight='bold')
    plt.title(f'Charging Demand',fontsize=25,weight='bold')
    
    ax.set_xlim(7*60, 21*60)  # X-axis limits from 7:00 to 21:00 (in minutes)
    ax.set_xticks(range(7*60, 22*60, 60))  # Every hour from 7:00 to 21:00 (in minutes)
    ax.set_xticklabels([f'{hour:02d}:00' for hour in range(7, 22)],fontsize=18)

    ax.set_ylim(0,max(charging_counts)+20)
    ax.tick_params(axis='y', labelsize=18) 

    plt.tight_layout()
    plt.show()



display_demand(schedules_1)


# %%
