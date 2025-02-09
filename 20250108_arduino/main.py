


#%%

import serial
import time

# Configure the serial port (replace 'COM3' with your serial port)
ser = serial.Serial('/dev/cu.usbmodem31201', 9600, timeout=1)

# Give some time for the serial connection to initialize
time.sleep(2)

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(line)
except KeyboardInterrupt:
    print("Serial communication stopped.")
finally:
    ser.close()

# %%
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Configure the serial port (replace 'COM3' with your serial port)
ser = serial.Serial('COM3', 9600, timeout=1)

# Parameters
N = 10000  # Maximum number of points to display

# Initialize data structures
xdata = deque(maxlen=N)
ydata = deque(maxlen=N)

# Initialize plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')

# Set up the plot limits and labels
ax.set_xlim(0, N)
ax.set_ylim(0, 1023)  # Adjust the y-axis limit based on your data range
ax.set_xlabel('Time')
ax.set_ylabel('Sensor Value')
ax.set_title('Real-Time Serial Data Plot')

# Update function for animation
def update(frame):
    while ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').strip()
            value = int(data)
            xdata.append(frame)
            ydata.append(value)
        except ValueError:
            pass
    line.set_data(range(len(xdata)), ydata)
    ax.relim()
    ax.autoscale_view()
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, interval=100)

# Show plot
plt.show()

# Close the serial port when done
ser.close()
