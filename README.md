# 1) Clone and enter the project
git clone https://github.com/samarth126/PID_to_LSTM.git cryo-lstm-pid
cd cryo-lstm-pid

# 2) Create & activate a virtual environment (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

#    (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) (Optional) Generate training data & train the model
#    This will create PID_train_data.csv and lstm_control.h5
python main1.py

# 5) Run the GUI inference app (connects to local sensors)
python onepy
