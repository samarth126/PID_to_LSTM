# LSTM-PID Cryogenic Temperature Control

This project implements a **PID + LSTM hybrid controller** for maintaining temperature in a **cryogenic liquid drum**.  
It is based on a research paper and demonstrates how **deep learning** can augment classical **PID control** for nonlinear and time-dependent dynamics.

---

## 🚀 Project Workflow

1. **Simulation & Training** (`main1.py`)  
   - Simulates a TC Labs cryogenic process.  
   - Collects data (`PID_train_data.csv`).  
   - Trains an LSTM controller.  
   - Saves the model weights (`lstm_control.h5`).  

2. **Inference & Control** (`onepy.py`)  
   - Loads the trained LSTM model.  
   - Launches a **Tkinter GUI**.  
   - Connects to local sensors/actuators.  
   - Maintains system temperature using the PID+LSTM controller.  

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/samarth126/PID_to_LSTM.git
cd PID_to_LSTM
