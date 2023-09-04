

#%%

import pandas as pd
file_path = 'loc.csv'
df = pd.read_csv(file_path, header=None, names=['lat', 'lng'])

#%% plot

from ipyleaflet import *
import random

m = Map(center=(df.lat[0],df.lng[0]), zoom=10, scroll_wheel_zoom=True)
m.add_control(FullScreenControl())

# %%
def random_color(feature):
    return {
        'color': 'black',
        'fillColor': random.choice(['red', 'yellow', 'green', 'orange']),
    }

icon_house = AwesomeIcon(name='home', marker_color='green')

t_drive=5400

for ii in range(len(df.lat)):
    
    with open(f'{t_drive}_{ii}.geojson', 'r') as json_file:
        data = json.load(json_file)
    geo_json = GeoJSON(
        data=data,
        style={
            'opacity': 1, 'dashArray': '9', 'fillOpacity': 0.2, 'weight': 1
        },
        hover_style={
            'color': 'white', 'dashArray': '0', 'fillOpacity': 0.5
        },
        style_callback=random_color
    )
    m.add_layer(geo_json)


    marker = Marker(location=(df.lat[ii], df.lng[ii]), draggable=False, icon=icon_house)
    m.add_layer(marker)

# add lakes
icon_lake = AwesomeIcon(name='fa-tint')
file_path = 'lakes.csv'
df_lakes = pd.read_csv(file_path, header=None, names=['lat', 'lng', 'name'])
for ii in range(len(df_lakes.lat)):
    marker = Marker(location=(df_lakes.lat[ii], df_lakes.lng[ii]), 
        draggable=False, icon=icon_lake, name=df_lakes.name[ii])
    m.add_layer(marker)



# %%
m

# %%
# %%
