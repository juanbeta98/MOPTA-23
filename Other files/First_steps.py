#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from gurobipy import *
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import truncnorm, bernoulli # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
from parameters import *

#%% Visual representation 

# Plot of vehicle's initial posiitons
img = plt.imread("Pennsylvania_Population.png")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-10, 305, -20, 165])
ax.scatter(coor[0], coor[1], color = 'purple', edgecolor = 'black', s = 5)
plt.title("Vehicle's locations"); plt.xlabel('lon'); plt.ylabel('lat')
plt.savefig('map', dpi = 600)


#%% Visualizing the probability function for rechaging 
x = np.linspace(20,250,1000)
prob = [np.exp(-0.012**2*(i-20)**2) for i in x]

plt.plot(x,prob)
plt.title('Probability of charging')
plt.xlabel("Vehicle's range")
plt.ylabel('Probability')
#plt.plot(89.39255074,0.5,color="red",marker="o")
#plt.text(x=89.39255074+1,y=0.5+0.1,s=f"Every vehicle with less than 90 miles\nof available range will recharge.")


#%%

indices = random.sample(range(len(coor)), 12)
sample_loc = coor.iloc[indices]
sample_loc.reset_index(drop=True, inplace=True)
coor.drop(coor.index[indices],inplace=True)
coor.reset_index(drop=True, inplace=True)

loc, scale, min_v, max_v= 100, 50, 20, 250
a, b = (min_v - loc) / scale, (max_v - loc) / scale

def feasibility_vehic_to_locations(df_vehic,v,r, df_loc):
    distances = [euclidean((df_vehic.loc[v,0], df_vehic.loc[v,1]), (df_loc.loc[l,0], df_loc.loc[l,1])) for l in range(df_loc.shape[0])]
    return sum([1 if r/d >= 1 else 0 for d in distances])

feasible = 0; sample_vehic = pd.DataFrame({0:[], 1:[], "range":[]})
while feasible < 100:
    index = random.randint(0,coor.shape[0]-1)
    range_real = truncnorm.rvs(a = a, b = b, loc = loc, scale = scale, size = 1)
    if feasibility_vehic_to_locations(coor,index,range_real,sample_loc) > 0:
        feasible += 1
        sample_vehic = pd.concat([sample_vehic,pd.DataFrame({0:coor.loc[index,0], 1:coor.loc[index,1], "range":[range_real]},index=[0])])
        sample_vehic.reset_index(drop=True, inplace=True)
        coor.drop(coor.index[index],inplace=True)
        coor.reset_index(drop=True, inplace=True)




#%%

V = range(len(sample_vehic))
S = range(len(sample_loc))

d = distance_matrix(sample_vehic,sample_loc)
r = {v:sample_vehic.loc[v,"range"][0] for v in V}
cf = 5500
cd = 0.041
cw = 0.0388



#%%

m = Model("MOPTA v1")

x = {s:m.addVar(name=f"x_{s}",vtype=GRB.BINARY) for s in S}
y = {s:m.addVar(name=f"y_{s}",vtype=GRB.INTEGER) for s in S}
z = {(v,s):m.addVar(name=f"z_{v,s}",vtype=GRB.BINARY) for v in V for s in S}

for s in S:
    # Only installs chargers in open stations
    m.addConstr(y[s] >= x[s])
    m.addConstr(y[s] <= u*x[s])
    # Maximum queue at each station
    m.addConstr(quicksum(z[v,s] for v in V) <= q*y[s])

for v in V:
    # Vehicle covering
    m.addConstr(quicksum(z[v,s] for s in S) == 1)
    for s in S:
        # Each vehicle can only be assigned to a station for which it has enough range to reach
        m.addConstr(d[v,s]*z[v,s] <= r[v])

# Number of stations to open
m.addConstr(quicksum(x[s] for s in S) == 7)

install_cost = quicksum(c_f*y[s] for s in S)
driving_cost = quicksum(c_d*d[v,s]*z[v,s] for v in V for s in S)
charging_cost = quicksum(c_w*(250-(r[v]-d[v,s]))*z[v,s] for v in V for s in S)
m.setObjective(install_cost + driving_cost + charging_cost)

m.update()
m.optimize()

try:
    sample_loc["open"] = [round(x[s].X) for s in S]
    sample_loc["chargers"] = [round(y[s].X) for s in S]
    sample_vehic["station"] = [s for v in V for s in S if z[v,s].X > 0.5]
except:
    print("Infeasible")

#%%


plt.scatter(sample_vehic[0], sample_vehic[1], color = 'purple', edgecolor = 'black', label="vehicle")
plt.scatter(sample_loc[0], sample_loc[1], color = 'goldenrod', edgecolor = 'black', label="unopened stations")
plt.scatter(sample_loc[sample_loc["open"]==1][0],sample_loc[sample_loc["open"]==1][1],label="opened stations",color="red")

plt.legend(loc="upper left")
plt.title('Vehicle locations'); plt.xlabel('x'); plt.ylabel('y')
plt.show()



# %%
