

#%%
import pandas as pd
file_path = 'loc.csv'
df = pd.read_csv(file_path, header=None, names=['lat', 'lng'])



# %%
from helpers_TT import get_geoJSON
import time

for ii in range(len(df.lat)):
    ii
    get_geoJSON(lat = df.lat[ii], lng = df.lng[ii], filename=f"{ii}.geojson")
    time.sleep(1)


# %%
