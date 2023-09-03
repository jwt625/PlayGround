

#%%

import pandas as pd
file_path = 'loc.csv'
df = pd.read_csv(file_path, header=None, names=['lat', 'lng'])

#%% plot

from ipyleaflet import *
import random

m = Map(center=(df.lat[0],df.lng[0]), zoom=10)


# %%
def random_color(feature):
    return {
        'color': 'black',
        'fillColor': random.choice(['red', 'yellow', 'green', 'orange']),
    }

for ii in range(7):
    with open(f'{ii}.geojson', 'r') as json_file:
        data = json.load(json_file)
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
    marker = Marker(location=(df.lat[ii], df.lng[ii]))
    m.add_layer(marker)

# %%
# %%


