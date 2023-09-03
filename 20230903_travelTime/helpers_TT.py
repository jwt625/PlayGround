
# travel time helpers

#%% using POST instead of the python SDK
from datetime import datetime
from traveltimepy import Driving, Coordinates, TravelTimeSdk
import requests
import json

def get_geoJSON(lat = 38.9451408, lng = -120.7226249,
                travel_time = 3600, transp_type = "driving",
                filename = "shape.geojson"):
    with open('apikey.json', 'r') as json_file:
        data = json.load(json_file)
    sdk = TravelTimeSdk(data['APP_ID'], data['APP_KEY'])
    path = "time-map"
    url = f"https://{sdk._sdk_params.host}/v4/{path}"
    
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

    with open(filename, "w") as text_file:
        text_file.write(response.text)

