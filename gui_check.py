import tkinter as tk
from tkinter import Label, Entry, Button
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# Define initial PID parameters
Kc_initial = 6.0
tauI_initial = 75.0
tauD_initial = 0.0

# Function to update PID parameters from user input
def update_data():
    pass

# Create a Tkinter window for input
root = tk.Tk()
root.title(" Enter Set point values")
root.geometry("500x150")

# Label and Entry widgets for PID parameters
Label(root, text="SP").grid(row=0, column=0)
Kc_entry = Entry(root)
Kc_entry.grid(row=0, column=1)
Kc_entry.insert(0, str(Kc_initial))

Label(root, text="temp").grid(row=1, column=0)
tauI_entry = Entry(root)
tauI_entry.grid(row=1, column=1)
tauI_entry.insert(0, str(tauI_initial))

Label(root, text="tauD:").grid(row=2, column=0)
tauD_entry = Entry(root)
tauD_entry.grid(row=2, column=1)
tauD_entry.insert(0, str(tauD_initial))

# Button to update PID parameters
update_button = Button(root, text="Update setpoints", command=update_data)
update_button.grid(row=3, columnspan=2)


