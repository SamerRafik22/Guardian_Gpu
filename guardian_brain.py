import time
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Arwa was here
# Khalid was here

LOG_DIR = r"d:\Guardian_Gpu\build\Release"
MODEL_PATH = "guardian_model.pkl"

class LogStreamer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.current_file = None
        self.file_handle = None
        self.last_pos = 0

    def get_latest_log(self):
        # Find latest CSV file
        files = glob.glob(os.path.join(self.log_dir, "gpu_log_*.csv"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def stream_lines(self):
        # 1. Open latest file
        latest = self.get_latest_log()
        if not latest:
            return []

        if latest != self.current_file:
            print(f"[Streamer] Switching to new log: {latest}")
            if self.file_handle:
                self.file_handle.close()
            self.current_file = latest
            self.file_handle = open(self.current_file, 'r')
            # Seek to end initially? No, we might want to read valid data if we just started
            # But for "Live" feel, let's process from NOW.
            # Actually, for the first run, let's read the last 100 lines for context?
            # Simpler: Just read from end for now.
            self.file_handle.seek(0, 2) 

        # 2. Read new lines
        lines = self.file_handle.readlines()
        return lines

class GuardianBrain:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = ['GPU_TIME_MS', 'GPU_PACKET_COUNT'] 
        self.knowledge = KnowledgeBank()
        # Dynamic Memory
        self.history_buffer = [] 
        self.max_history = 1000 # Learn from last 1000 events
        self.learn_counter = 0

    def train_initial(self, log_dir_or_file):
        # Handle Directory
        if os.path.isdir(log_dir_or_file):
            files = glob.glob(os.path.join(log_dir_or_file, "gpu_log_*.csv"))
            if not files:
                 print("[Brain] No data found. Starting fresh.")
                 self.is_trained = True
                 return
            # Load last 5 logs
            df_list = []
            for f in files[-5:]:
                try:
                    df = pd.read_csv(f)
                    # Check columns
                    for col in self.feature_cols:
                        if col not in df.columns: df[col] = 0
                    df_list.append(df[self.feature_cols])
                except: pass
            
            if not df_list: return
            X = pd.concat(df_list).values
            
        else: # Single File
            if not os.path.exists(log_dir_or_file): return
            X = pd.read_csv(log_dir_or_file)[self.feature_cols].fillna(0).values

        print(f"[Brain] Training on {len(X)} historic events...")
        
        # Bootstrap History
        self.history_buffer = X[-self.max_history:].tolist()
        
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X))
        self.is_trained = True
        print("[Brain] Model Trained. Dynamic Learning Active.")

    def retrain_dynamic(self):
        if len(self.history_buffer) < 50: return
        
        # Fast Retrain on Window
        try:
            X = np.array(self.history_buffer)
            self.scaler.fit(X) # Re-center on recent data
            self.model.fit(self.scaler.transform(X))
            # print("[Brain] Evolved!") # Debug
        except:
            pass

    def predict(self, line_data):
        # Update History
        row = [line_data.get(col, 0) for col in self.feature_cols]
        self.history_buffer.append(row)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
            
        # Continuous Learning: Retrain every 50 events (~5 seconds)
        self.learn_counter += 1
        if self.learn_counter >= 50:
            self.retrain_dynamic()
            self.learn_counter = 0

        if not self.is_trained:
            return 0, 0, []

        # 1. Check Knowledge Bank
        known_labels = self.knowledge.is_known(row)
        
        # 2. ML Prediction
        try:
            X = np.array([row])
            X_scaled = self.scaler.transform(X)
            score = self.model.predict(X_scaled)[0]
            severity = self.model.score_samples(X_scaled)[0]
        except:
             score = -1 
             severity = -1.0

        return score, severity, known_labels

# --- Helper for Printing ---
class SessionTracker:
    def __init__(self):
        self.stats = {} # Dynamic
        self.total = 0
        self.matches = 0
        self.alerts = 0
        self.last_print = time.time()

    def update(self, category):
        if category not in self.stats:
            self.stats[category] = 0
        self.stats[category] += 1
        self.total += 1

    def record_match(self):
        self.matches += 1

    def record_alert(self):
        self.alerts += 1

    def print_summary(self):
        if self.total == 0: return

        # Print every 2 seconds for better feedback
        if time.time() - self.last_print < 2:
            return

        # Calculate Health (100% - Anomaly Rate)
        health = 100.0
        if self.total > 0:
            health = 100.0 - (self.alerts / self.total * 100.0)

        print(f"\r[Summary] Health: {health:.1f}% | KB Matches: {self.matches} | ", end="")
        for cat, count in self.stats.items():
            pct = (count / self.total) * 100
            if pct > 1:
                print(f"{cat}: {pct:.1f}% | ", end="")
        
        self.last_print = time.time()


def classify_activity(data):
    # Heuristic Classification Rules
    time_ms = float(data.get('GPU_TIME_MS', 0))
    count = float(data.get('GPU_PACKET_COUNT', 0))
    
    # 1. Suspicious Copy (High Traffic, Zero Compute)
    if count > 200 and time_ms < 1.0:
        return "SUSPICIOUS_COPY"
    
    # 2. Gaming (High Traffic AND Significant Compute)
    if count > 100 and time_ms > 15.0:
        return "Gaming"
        
    # 3. Compute / Mining (High Compute, Low Traffic)
    if time_ms > 100.0: 
        return "Compute"

    # 4. Standard Desktop 3D (High Traffic, Low Compute)
    if count > 50:
        return "3D/UI Activity"
        
    return "Idle"    
    

def parse_line(line):
    # CSV Format: TIMESTAMP,PID,NAME,MEM_MB,PWR_W,GPU_TIME_MS,GPU_PACKET_COUNT,NET_TX,NET_RX
    try:
        parts = line.strip().split(',')
        if len(parts) < 7: return None # Bad line
        
        # Check if it's a header
        if "TIMESTAMP" in parts[0]: return None
        
        has_net = (len(parts) >= 9)
        
        obj = {}
        obj['PID'] = parts[1]
        obj['NAME'] = parts[2].replace('"', '')
        
        if has_net:
            obj['NET_RX'] = float(parts[-1])
            obj['NET_TX'] = float(parts[-2])
            obj['GPU_PACKET_COUNT'] = float(parts[-3])
            obj['GPU_TIME_MS'] = float(parts[-4])
            obj['PWR_W'] = float(parts[-5])
            obj['MEM_MB'] = float(parts[-6]) 
        else:
            obj['NET_RX'] = 0
            obj['NET_TX'] = 0
            obj['GPU_PACKET_COUNT'] = float(parts[-1])
            obj['GPU_TIME_MS'] = float(parts[-2])
            obj['PWR_W'] = float(parts[-3])
            obj['MEM_MB'] = float(parts[-4])

        return obj
    except Exception as e:
        return None

def main():
    print("----------------------------------------------------------------")
    print("   GUARDIAN BRAIN - Live Anomaly Detection System")
    print("----------------------------------------------------------------")
    
    brain = GuardianBrain()
    brain.train_initial(LOG_DIR)
    
    streamer = LogStreamer(LOG_DIR)
    session = SessionTracker()
    
    print("[Streamer] Watching for live events...")
    
    try:
        while True:
            lines = streamer.stream_lines()
            for line in lines:
                data = parse_line(line)
                if data:
                    # 1. Classify Activity
                    category = classify_activity(data)
                    session.update(category)
                    
                    # 2. Check Anomaly
                    score, severity, known_labels = brain.predict(data)
                    
                    if category == "SUSPICIOUS_COPY":
                        score = -1
                        severity = -0.9999 # Maximum Severity
                        known_labels = [] # Force it to be unknown to KB users

                    if score == -1:
                        if known_labels:
                             # Suppress Alert & Record Match
                             session.record_match()
                        else:
                            # Anomaly!!
                            session.record_alert()
                            print(f"\n!!! ANOMALY DETECTED !!! [Severity: {severity:.4f}]")
                            print(f"    PID: {data['PID']} ({data['NAME']})")
                            print(f"    Activity: {category}")
                            print(f"    GPU: {data['GPU_TIME_MS']}ms | Counts: {data['GPU_PACKET_COUNT']}")
                            print(f"    Net: Tx {data['NET_TX']} | Rx {data['NET_RX']}")
                            print("")
            
            # Print Summary
            session.print_summary()
                        
            time.sleep(0.1) # 100ms latency
            
    except KeyboardInterrupt:
        print("Stopping Brain.")

if __name__ == "__main__":
    main()
