
#%%
import asyncio
from datetime import datetime
import json


# have to use this to get rid of "RuntimeError: asyncio.run() cannot be called from a running event loop"
import nest_asyncio

nest_asyncio.apply() 

#%% test example
with open('apikey.json', 'r') as json_file:
    data = json.load(json_file)

#%%
from traveltimepy import Driving, Coordinates, TravelTimeSdk

async def main():
    sdk = TravelTimeSdk(data['APP_ID'], data['APP_KEY'])
    
    results = await sdk.time_map_async(
        coordinates=[Coordinates(lat=51.507609, lng=-0.128315), 
                     Coordinates(lat=51.517609, lng=-0.138315)],
        arrival_time=datetime.now(),
        transportation=Driving()
    )
    print(results)

asyncio.run(main())


#%% try to get an isochrone
lat = 38.9451408
lng = -120.7226249

from traveltimepy import Driving, Coordinates, TravelTimeSdk


async def main():
    sdk = TravelTimeSdk(data['APP_ID'], data['APP_KEY'])
    
    results = await sdk.time_map_async(
        coordinates=[Coordinates(lat=lat, lng=lng)],
        arrival_time=datetime.now(),
        transportation=Driving(),
        travel_time=3600
    )
    print(results)
    return results

res = asyncio.run(main())

#%%
res[0].shapes

# %% testing ipyleaflet
from ipyleaflet import *
m = Map(center=(50, 0))

# %%
