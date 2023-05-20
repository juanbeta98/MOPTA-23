#%%
import pandas as pd
import os 

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

    if (long <= 290) & (lat <= 150) & (lat >= 0) & (long >= 0):
        latitudes.append(stations['Latitude'][i])
        longitudes.append(stations['Longitude'][i])

dd = {'latitude':latitudes, 'longitude':longitudes}
df = pd.DataFrame(data = dd)
df.to_excel('PROVISIONAL.xlsx')

#%%