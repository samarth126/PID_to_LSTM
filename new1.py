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
run_time = 14.0

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
end = 100
start = 15

Tsp[start:end] = 34

# leave last 120 seconds as room temp
Tsp[-120:] = Tsp[0]
plt.plot(Tsp)
plt.show()



j=0

# Run test
with TCLab() as lab:
    # Find current T1, T2
    print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
    print('Temperature 2: {0:0.2f} °C'.format(lab.T2))
    
    start_time = 0
    prev_time = 0
    for i,t in enumerate(tclab.clock(loops)):
        if j == 0:
            set_point = int(input("input wait.."))
            Tsp[i:i+500] = set_point
            j = 500
            print(i,j)
            plt.plot(Tsp)
            plt.show()
        else:
            j = j-1
        tm[i] = t
        dt = t - prev_time

        # Read temperature (C)
        T1[i] = lab.T1

        # Run LSTM model to get Q1 value for control
        if i >= window:
            # Load data for model
            T1_m = T1[i-window:i]
            Tsp_m = Tsp[i-window:i]
            # Predict and store LSTM value for comparison
            Qlstm[i] = lstm(T1_m,Tsp_m)

        # Write heater output (0-100)
        lab.Q1(Qlstm[i])

        prev_time = t

plt.figure(figsize=(10,4))
plt.plot(Tsp[:i],'k-',label='SP $(^oC)$')
plt.plot(T1[:i],'r-',label='$T_1$ $(^oC)$')
plt.plot(Qlstm[:i],'g-',label='$Q_{LSTM}$ (%)')
plt.legend(fontsize=14)
plt.ylim((0,100))
plt.xlabel('Time (s)',size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid()
plt.show()





