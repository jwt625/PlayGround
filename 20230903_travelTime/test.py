
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
shape = res[0].shapes[0]




#%% using POST instead of the python SDK
from datetime import datetime

import requests
import json

lat = 38.9451408
lng = -120.7226249
sdk = TravelTimeSdk(data['APP_ID'], data['APP_KEY'])
path = "time-map"
url = f"https://{sdk._sdk_params.host}/v4/{path}"
transp_type = "driving"
travel_time = 3600

# url = "https://api.traveltimeapp.com/v4/time-map"

payload = json.dumps({
    "departure_searches": [
        {
            "id": "public transport from Trafalgar Square",
            "coords": {"lat": lat, "lng": lng},
            "transportation": {"type": transp_type},
            "departure_time": datetime.utcnow().isoformat(),
            "travel_time": travel_time
        }
    ]
})

headers = {
    'Host': 'api.traveltimeapp.com',
    'Content-Type': 'application/json',
    'Accept': 'application/geo+json',
    'X-Application-Id': sdk._app_id,
    'X-Api-Key': sdk._api_key
}

response = requests.request("POST", url, headers=headers, data=payload)

with open("shape.geojson", "w") as text_file:
    text_file.write(response.text)




# %% testing ipyleaflet
from ipyleaflet import *
m = Map(center=(50, 0))

with open('shape.geojson', 'r') as json_file:
    data = json.load(json_file)

# %%

import random
def random_color(feature):
    return {
        'color': 'black',
        'fillColor': random.choice(['red', 'yellow', 'green', 'orange']),
    }

geo_json = GeoJSON(
    data=data,
    style={
        'opacity': 1, 'dashArray': '9', 'fillOpacity': 0.1, 'weight': 1
    },
    hover_style={
        'color': 'white', 'dashArray': '0', 'fillOpacity': 0.5
    },
    style_callback=random_color
)
m.add_layer(geo_json)