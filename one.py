import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm # Progress bar
import gc
# for scaling

# For scaling, feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split 


# For LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from keras.models import load_model

# For TCLab
import tclab
from matplotlib.animation import FuncAnimation

import tkinter as tk
from tkinter import Label, Button, Entry

speedup = 10000
TCLab = tclab.setup(connected=False, speedup = speedup)


new_model = tf.keras.models.load_model('lstm_control.h5')

new_df = pd.read_csv('PID_train_data.csv')


print(new_df)

# Scale data
X = new_df[['Tsp1','err']].values
y = new_df[['Q1']].values
s_x = MinMaxScaler()
Xs = s_x.fit_transform(X)

s_y = MinMaxScaler()
ys = s_y.fit_transform(y)
window = 15


# Show the model architecture
# print(new_model.summary())


# LSTM controller code
def lstm(T1_m, Tsp_m):
    # Calculate error (necessary feature for LSTM input)
    err = Tsp_m - T1_m




    # Format data for LSTM input
    X = np.vstack((Tsp_m,err)).T
    Xs = s_x.transform(X)
    Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))

    # Predict Q for controller and unscale
    Q1c_s = new_model.predict(Xs)
    Q1c = s_y.inverse_transform(Q1c_s)[0][0]

    # Ensure Q1c is between 0 and 100
    Q1c = np.clip(Q1c,0.0,100.0)
    return Q1c





# Define a function to save data to a CSV file
def save_data_to_csv(i, i_values, Tsp_values, T1_values, Qlstm_values):
    data = {
        'i_values': i_values,
        'Tsp_values': Tsp_values,
        'T1_values': T1_values,
        'Qlstm_values': Qlstm_values
    }

    csv_file = "output.csv"

    # If the file doesn't exist, create it and write the header
    if not os.path.exists(csv_file):
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"Data has been saved to {csv_file}")
    else:
        # If the file exists, append data to it
        df = pd.read_csv(csv_file)
        new_data = pd.DataFrame(data)
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(csv_file, index=False, mode='w')
        print(f"Data has been appended to {csv_file}")
    
    # Clear the DataFrame to release memory
    df = None
    gc.collect()
    return







# Run time in minutes
run_time = 1440.0

# Number of cycles
loops = int(60.0*run_time)

# arrays for storing data
T1 = np.zeros(loops) # measured T (degC)
Qpid = np.zeros(loops) # Heater values for PID controller
Qlstm = np.zeros(loops) # Heater values for LSTM controller
tm = np.zeros(loops) # Time

# Temperature set point (degC)
with TCLab() as lab:
    Tsp = np.ones(loops) * lab.T1

# vary temperature setpoint
end = window + 15 # leave 1st window + 15 seconds of temp set point as room temp

Tsp[end:] = 40
# leave last 120 seconds as room temp
Tsp[-120:] = Tsp[0]
plt.plot(Tsp)
plt.show()

itrator= 0

manual_flag = True
manual_val= 10
sp_g = 40

crr_LSTM_Q = 0
crr_tem = 0
crr_tsp = 0

def mann_update():
    global manual_flag
    global manual_val
    manual_flag = True
    manual_heating = manual_heat.get()
    manual_val = int(manual_heating)
    manual.configure(bg="green")
    auto.configure(bg="red")


def lstm_auto():
    global manual_flag
    manual_flag = False
    manual.configure(bg="red")
    auto.configure(bg="green")
    

# Function to update Tsp values
def update_Tsp():
    global itrator
    global Tsp
    global sp_g
    end = window + 15
    start = end
    set_point = sp.get()
    sp_g = int(set_point)
    Tsp[itrator+1:] = sp_g
    print("Set Point:", set_point)

# Create a Tkinter window
root = tk.Tk()
root.title("Temperature Setpoint (Tsp)")

label = tk.Label(root, text="Set Point:")
label.grid(row=2, column=0)
sp = tk.Entry(root)
sp.grid(row=2, column=2)

sp.insert(0, sp_g)

# Create a button to update Tsp values
update_button = tk.Button(root, text="Update Tsp", command=update_Tsp)
update_button.grid(row=3, column=0, columnspan=2)

# Label and Entry for Manual Value
manual_label = tk.Label(root, text="Manual Value:")
manual_label.grid(row=4, column=0)
manual_heat = tk.Entry(root)
manual_heat.grid(row=4, column=2)

manual_heat.insert(0, manual_val)

manual = tk.Button(root, text="Manual Override", command=mann_update)
auto = tk.Button(root, text="LSTM auto", command=lstm_auto)

manual.configure(bg="green")
auto.configure(bg="red")

manual.grid(row=5, column=0, columnspan=2)
auto.grid(row=5, column=1, columnspan=2)

# Create an Entry widget for 'y'
crr_lstm_label = tk.Label(root, text=f"y = {crr_LSTM_Q}")
crr_lstm_label.grid(row=7, column=0)

crr_man_label = tk.Label(root, text=f"y = {manual_val}")
crr_man_label.grid(row=8, column=0)

crr_tem_label = tk.Label(root, text=f"y = {crr_tem}")
crr_tem_label.grid(row=9, column=0)

crr_tsp_label = tk.Label(root, text=f"y = {crr_tsp}")
crr_tsp_label.grid(row=10, column=0)

# Create a figure and axes for the real-time plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_ylim((0, 100))
ax.set_xlabel('Time (s)', size=14)
ax.tick_params(axis='both', which='both', labelsize=12)

# Initialize data arrays for real-time plot
i_values = []
Tsp_values = []
T1_values = []
Qlstm_values = []

# Create empty lines for the plot
line_sp, = ax.plot([], 'k-', label='SP $(^oC)$')
line_t1, = ax.plot([], 'r-', label='$T_1$ $(^oC)$')
line_lstm, = ax.plot([], 'g-', label='$Q_{LSTM}$ (%)')
ax.legend(loc='upper right', fontsize=14)

# Initialize the plot
plt.ion()  # Turn on interactive mode
plt.show()

# Run test
with TCLab() as lab:
    # Find current T1, T2
    print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
    print('Temperature 2: {0:0.2f} °C'.format(lab.T2))

    start_time = 0
    prev_time = 0

    for i, t in enumerate(tclab.clock(loops)):
        itrator= i
        tm[i] = t
        dt = t - prev_time

        # Read temperature (C)
        T1[i] = lab.T1

        # Run LSTM model to get Q1 value for control
        if i >= window:
            # Load data for model
            T1_m = T1[i - window:i]
            Tsp_m = Tsp[i - window:i]
            # Predict and store LSTM value for comparison
            Qlstm[i] = lstm(T1_m, Tsp_m)

        crr_LSTM_Q = round(Qlstm[i], 2)
        crr_tem = round(T1[i], 2)
        crr_tsp = round(Tsp[i], 2)

        # Write heater output (0-100)
        print(manual_flag)

        #setting changing values in GUI
        crr_lstm_label.config(text=f"Current LSTM Q= {crr_LSTM_Q}") 
        crr_man_label.config(text=f"Current Manual Q= {manual_val}") 
        crr_tem_label.config(text=f"Current PV= {crr_tem}") 
        crr_tsp_label.config(text=f"Current Tsp= {crr_tsp}") 


        if manual_flag == True:
            lab.Q1(manual_val)
            Qlstm_values.append(manual_val)
        else:
            lab.Q1(Qlstm[i])
            Qlstm_values.append(Qlstm[i])

        # Update plot data
        i_values.append(i)
        Tsp_values.append(Tsp[i])
        T1_values.append(T1[i])
        

        # Update plot lines
        line_sp.set_data(i_values, Tsp_values)
        line_t1.set_data(i_values, T1_values)
        line_lstm.set_data(i_values, Qlstm_values)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.flush_events()  # Update the plot

        prev_time = t

        if i % 3600 == 0 and i != 0:
            # Save data every 3600 loops
            save_data_to_csv(i, i_values, Tsp_values, T1_values, Qlstm_values)

            # Clear the lists to free up memory
            i_values.clear()
            Tsp_values.clear()
            T1_values.clear()
            Qlstm_values.clear()


# Turn off interactive mode after the loop
plt.ioff()
plt.show()


# data = {
#     'i_values': i_values,
#     'Tsp_values': Tsp_values,
#     'T1_values': T1_values,
#     'Qlstm_values': Qlstm_values
# }

# df = pd.DataFrame(data)

# # Define the CSV file name
# csv_file = "output.csv"

# # Save the DataFrame to a CSV file
# df.to_csv(csv_file, index=False)

# print(f"Data has been saved to {csv_file}")



