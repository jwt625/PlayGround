
#%%
import asyncio
from datetime import datetime
import json


#%%
with open('apikey.json', 'r') as json_file:
    data = json.load(json_file)

#%%
from traveltimepy import Driving, Coordinates, TravelTimeSdk

async def main():
    sdk = TravelTimeSdk(data.APP_ID, data.APP_KEY)
    
    results = await sdk.time_map_async(
        coordinates=[Coordinates(lat=51.507609, lng=-0.128315), Coordinates(lat=51.517609, lng=-0.138315)],
        arrival_time=datetime.now(),
        transportation=Driving()
    )
    print(results)

asyncio.run(main())

# %%
