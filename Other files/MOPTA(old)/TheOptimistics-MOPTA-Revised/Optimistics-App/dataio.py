import json

def dataFromDict(dic:dict):
    """ takes the request.get_json dictionary and returns the input for the H-SARA problem"""

    data = dic

    maxTime = int(data['maxTime'])
    costFixed = data['costFixed']
    costTime = data['costTime']
    costOvertime = data['costOvertime']
    alpha = data['alpha']
    mean_service_time = data['serviceTime']
    randomInstance = data['RamdonInstance']
    numCustomers = int(data['numCustomers'])
    lat = data['latitude']
    lon = data['longitude']
    prob = data['probability']
    hour = str(data['hour'])
    minute = str(data['minute'])
    t_max = data['tmax']
    sol_method = data['sol_method']
    
    return maxTime, costFixed, costTime, costOvertime, alpha, mean_service_time,randomInstance, numCustomers, lat, lon, prob, hour, minute, t_max, sol_method