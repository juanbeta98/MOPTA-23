#%%
import pandas as pd
import os 
import pickle

path = os.getcwd()

stations = pd.read_csv(path+"/fuel_stations.csv")

northern = (-79.761960, 42.269385)
southern = (-76.9909,39.7198)
western = (-80.519400, 40.639400)
eastern = (-74.689603, 41.363559)


latitudes = list()
longitudes = list()

for i in stations.index:
    long = (stations["Longitude"][i]-western[0])*52-1
    lat  = (stations["Latitude"][i]-southern[1])*69-7

    if (long <= 290) & (lat <= 150) & (lat >= 0) & (long >= 0) and i+1:
        latitudes.append(stations['Latitude'][i])
        longitudes.append(stations['Longitude'][i])

file = open('../Results/Configurations/Open Stations/open_stations_18', 'rb')
open_stations = pickle.load(file)
file.close()
op_lat = list(); op_lon = list()
cl_lat = list(); cl_lon = list()
for i in range(len(longitudes)):
    if i+1 in open_stations:
        op_lat.append(latitudes[i]); op_lon.append(longitudes[i])
    else:
        cl_lat.append(latitudes[i]); cl_lon.append(longitudes[i])




# dd = {'latitude':op_lat, 'longitude':op_lon}
# df = pd.DataFrame(data = dd)
# df.to_excel('OPENEDPROVISIONALOPEN.xlsx')

ddd = {'latitude':cl_lat, 'longitude':cl_lon}
ddff = pd.DataFrame(data = ddd)
ddff.to_excel('PROVISIONALCLOSED.xlsx')

#%%
# %%
