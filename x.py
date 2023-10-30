import matplotlib.pyplot as plt
from datetime import datetime

# Create a list of time values in the "hour:minute:second" format as strings
time_strings = ['09:46:52', '10:30:15', '11:15:30', '12:00:00']

# Convert the time strings to numeric values representing time in seconds
time_values = [datetime.strptime(time_str, '%H:%M:%S') for time_str in time_strings]

# Create a list of values that correspond to your data points, e.g., y-values
y_values = [10, 20, 15, 30]

# Plot the data using the time values in seconds for the x-axis
plt.plot(time_values, y_values)

# Customize your plot as needed (e.g., labels, titles, etc.).
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.title('Time vs. Value')

# Show the plot
plt.show()