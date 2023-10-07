import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm # Progress bar

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

speedup = 100
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










# Run time in minutes
run_time = 60.0

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
while end <= loops:
    start = end
    # keep new temp set point value for anywhere from 4 to 10 min
    end += random.randint(240,600)
    Tsp[start:end] = random.randint(30,70)

# leave last 120 seconds as room temp
Tsp[-120:] = Tsp[0]
plt.plot(Tsp)
plt.show()










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

        # Write heater output (0-100)
        lab.Q1(Qlstm[i])

        # Update plot data
        i_values.append(i)
        Tsp_values.append(Tsp[i])
        T1_values.append(T1[i])
        Qlstm_values.append(Qlstm[i])

        # Update plot lines
        line_sp.set_data(i_values, Tsp_values)
        line_t1.set_data(i_values, T1_values)
        line_lstm.set_data(i_values, Qlstm_values)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.flush_events()  # Update the plot

        prev_time = t

# Turn off interactive mode after the loop
plt.ioff()
plt.show()