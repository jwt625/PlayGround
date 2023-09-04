

#%%
import pandas as pd
file_path = 'loc.csv'
df = pd.read_csv(file_path, header=None, names=['lat', 'lng'])



# %%
from helpers_TT import get_geoJSON
import time

# t_drive = 3600+1800
for t_drive in [3600, 5400, 7200]:
    # for ii in range(len(df.lat)):
    for ii in [len(df.lat)-1]:    # only adding the last one
        ii
        get_geoJSON(lat = df.lat[ii], lng = df.lng[ii],
                    filename=f"{t_drive}_{ii}.geojson", travel_time=t_drive
                    )
        time.sleep(1)


# %%
