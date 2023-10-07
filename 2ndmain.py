import os
import numpy as np
import random
import time
import tkinter as tk
from tkinter import Label, Button
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tclab

speedup = 100
TCLab = tclab.setup(connected=False, speedup=speedup)



window=15



# Run time in minutes
run_time = 60.0

# Number of cycles
loops = int(60.0 * run_time)

# Arrays for storing data
T1 = np.zeros(loops)  # Measured T (degC)
Qpid = np.zeros(loops)  # Heater values for PID controller
Qlstm = np.zeros(loops)  # Heater values for LSTM controller
tm = np.zeros(loops)  # Time

# Temperature set point (degC)
with TCLab() as lab:
    Tsp = np.ones(loops) * lab.T1

# Vary temperature setpoint
end = window + 15  # Leave 1st window + 15 seconds of temp set point as room temp
while end <= loops:
    start = end
    # Keep the new temp set point value for anywhere from 4 to 10 min
    end += random.randint(180, 300)
    Tsp[start:end] = random.randint(30, 70)

# Leave the last 120 seconds as room temp
Tsp[-120:] = Tsp[0]




# Function to update Tsp values
def update_Tsp():
    global Tsp
    end = window + 15
    start = end
    end += 300
    Tsp[start:end] = random.randint(30, 70)

# Create a Tkinter window
root = tk.Tk()
root.title("Temperature Setpoint (Tsp)")

# Create a figure and axes for the real-time plot
fig, ax = plt.subplots(figsize=(10, 4))
line_tsp, = ax.plot([], 'k-', label='SP $(^oC)$')
ax.legend(loc='upper right', fontsize=14)
ax.set_ylim((0, 100))
ax.set_xlabel('Time (s)', size=14)
ax.tick_params(axis='both', which='both', labelsize=12)

# Initialize data arrays for real-time plot
i_values = []
Tsp_values = []

def update_plot(i):
    i_values.append(i)
    Tsp_values.append(Tsp[i])
    line_tsp.set_data(i_values, Tsp_values)
    ax.relim()
    ax.autoscale_view()

# Create an animation that updates the plot every loop iteration
ani = FuncAnimation(fig, update_plot, frames=range(loops), repeat=False)

# Create a button to update Tsp values
update_button = Button(root, text="Update Tsp", command=update_Tsp)
update_button.pack()

# Show the animation (real-time plot)
plt.title("Temperature Setpoint (Tsp)")
plt.show()

