

#%%

import matplotlib.pyplot as plt
import numpy as np

plt.xkcd()  # Yes...
plt.plot(np.sin(np.linspace(0, 10)))
plt.title('Whoo Hoo!!!')



# %%
# # downloaded font at https://github.com/ipython/xkcd-font
import matplotlib.font_manager
matplotlib.font_manager._rebuild()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = "./registrations_per_date.csv"
df = pd.read_csv(file_path, parse_dates=['registration_date'])

# Ensure data is sorted by date
df = df.sort_values('registration_date')

# Compute cumulative sum
df['cumulative_registrations'] = df['num_registrations'].cumsum()

# Identify peaks (manually provided indices based on significant events)
peak_dates = ["2024-02-07", "2024-08-31", "2024-10-17", "2024-11-14"]
annotations = [
    "Bluesky opens to public",
    "Twitter/X banned in Brazil",
    "Twitter/X changes visibility policy",
    "2024 US Elections"
]

# Plot with xkcd style
plt.xkcd()
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary y-axis (daily registrations)
ax1.set_xlabel("Date")
ax1.set_ylabel("Registrations per day", color="tab:blue")
ax1.plot(df['registration_date'], df['num_registrations'], color="tab:blue", label="Daily Registrations")
ax1.tick_params(axis='y', labelcolor="tab:blue")

# Secondary y-axis (cumulative registrations)
ax2 = ax1.twinx()
ax2.set_ylabel("Cumulative Registrations", color="tab:red")
ax2.plot(df['registration_date'], df['cumulative_registrations'], color="tab:red", linestyle="--", label="Cumulative Registrations")
ax2.tick_params(axis='y', labelcolor="tab:red")

# Annotate peaks
for peak_date, note in zip(peak_dates, annotations):
    peak_idx = df[df['registration_date'] == peak_date].index
    if not peak_idx.empty:
        peak_idx = peak_idx[0]
        peak_x = df.loc[peak_idx, 'registration_date']
        peak_y = df.loc[peak_idx, 'num_registrations']
        ax1.annotate(note, xy=(peak_x, peak_y), xytext=(peak_x - pd.Timedelta(days=400), peak_y - 10000),
                     arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=14)

# Titles and layout
plt.title("Bluesky Registrations Over Time")
# fig.tight_layout()
plt.show()
# ax1.grid(True, linestyle="--", alpha=0.7)
# plt.grid(True)

# %%
