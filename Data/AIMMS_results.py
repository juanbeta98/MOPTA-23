#%%
# Modules
import pandas as pd
import os 
import pickle
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
print(f'{cont} unassigned profile stations')

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









#%%