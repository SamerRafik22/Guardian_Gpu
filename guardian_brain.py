import time
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import warnings
import atexit
import gzip
import json
import threading
import queue
from sklearn.neighbors import LocalOutlierFactor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

LOG_DIR = r"d:\Guardian_Gpu\build\Release"
MODEL_PATH = "brain_state.pkg"
VAULT_PATH = "guardian_vault.jsonl.gz"
KB_PATH = "knowledge_bank.json"

class KnowledgeBank:
    def __init__(self, kb_path=KB_PATH):
        self.kb_path = kb_path
        self.known_signatures = []
        self.load()

    def load(self):
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, 'r') as f:
                    self.known_signatures = json.load(f)
            except:
                self.known_signatures = []

    def is_known(self, vector, tolerance=0.1):
        if not self.known_signatures:
             return []
        
        # Simple Euclidean Logic for PoC
        try:
             vec = np.array(vector)
             for sig in self.known_signatures:
                 ref = np.array(sig['vector'])
                 dist = np.linalg.norm(vec - ref)
                 if dist < tolerance:
                     return [sig['label']]
        except: pass
        return []

class BackgroundAnalyzer(threading.Thread):
    def __init__(self, knowledge_bank, vault):
        super().__init__()
        self.kb = knowledge_bank
        self.vault = vault
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True # Auto-kill when main dies
        
    def run(self):
        print("[Background] Ghost Thread initialized. Waiting for cases...")
        while self.running:
            try:
                # Wait for suspect data (timeout allows check strictly for exit)
                priority, data_row, original_score = self.queue.get(timeout=1.0)
                
                # --- HEAVY COMPUTATION ZONE (Tier 4) ---
                # This runs in parallel and doesn't block the game
                
                # 1. Simulate Deep Analysis (e.g., LOF on Vault History)
                # In real prod, we would load 5000 pts from Vault here
                time.sleep(1.0) # Simulating "Thinking" time
                
                # 2. Logic: If it's ambiguous (-0.65), do we find neighbors?
                # For this implementation, we simulate accurate "Re-Classification"
                
                final_verdict = "SAFE"
                # If we find it in KnowledgeBank -> Safe
                if self.kb.is_known(data_row):
                    final_verdict = "SAFE"
                else:
                    # If density is low (Outlier) -> THREAT
                    # Using Mock LOF logic for now to respect stability
                    final_verdict = "ANALYZED_THREAT"

                # 3. Update System
                if final_verdict == "SAFE":
                    # Teach the Brain so main thread doesn't ask again
                    # Accessing KB needs to be thread-safe in prod, simple list append is atomic enough for Python GIL often
                    self.kb.known_signatures.append({'vector': data_row, 'label': 'SAFE (Bg Verified)'})
                    print(f"\n[Bg] CASE RESOLVED: {data_row} marked SAFE after deep analysis.")
                else:
                    print(f"\n[Bg] THREAT CONFIRMED: {data_row} | Score: {original_score}")

                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Bg] Error: {e}")

    def submit(self, data_row, score):
        self.queue.put((1, data_row, score))

class GuardianVault:
    def __init__(self, vault_path=VAULT_PATH):
        self.vault_path = vault_path

    def flush(self, data_chunk):
        """Compress and Append data to Vault"""
        if not data_chunk: return
        
        try:
            # Mode 'at' (Append Text) doesn't exist for gzip, so we use 'ab' (Append Binary)
            with gzip.open(self.vault_path, 'ab') as f:
                for row in data_chunk:
                    # Compact JSON string + Newline
                    line = json.dumps(row) + "\n"
                    f.write(line.encode('utf-8'))
            print(f"[Vault] Secured {len(data_chunk)} memories to Deep Storage.")
        except Exception as e:
            print(f"[Vault] Flush Failed: {e}")

    def audit(self, query_vector, tolerance=0.1):
        """Scan Vault for similar patterns (Double Check)"""
        if not os.path.exists(self.vault_path): return False
        
        match_found = False
        try:
            # We only read the last 5000 lines to avoid full scan lag
            # For PoC, we scan all (file is small)
            with gzip.open(self.vault_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # Check similarity: GPU_TIME and PACKET_COUNT
                        # Simple Euclidean check
                        dist = np.linalg.norm(np.array(query_vector) - np.array(record))
                        if dist < tolerance:
                            match_found = True
                            break
                    except: continue
        except Exception as e:
            print(f"[Vault] Audit Error: {e}")
            
        return match_found

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
        self.vault = GuardianVault()
        
        # Dynamic Memory
        self.history_buffer = [] 
        self.max_history = 10000 # [MODIFIED] Learn from last 10,000 events (~3 Hours)
        self.learn_counter = 0
        
        # Background Intelligence (Tier 4)
        self.ghost = BackgroundAnalyzer(self.knowledge, self.vault)
        self.ghost.start()
        
        # Load State if exists (Resurrection Layer)
        self.load_state()
        
        # Auto-Save on Exit
        atexit.register(self.save_state)

    def save_state(self):
        try:
            state = {
                'buffer': self.history_buffer,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            joblib.dump(state, MODEL_PATH, compress=3)
            print(f"[Brain] State Saved: {len(self.history_buffer)} events persisted.")
        except Exception as e:
            print(f"[Brain] Save Failed: {e}")

    def load_state(self):
        if not os.path.exists(MODEL_PATH):
            return
            
        try:
            print("[Brain] Resurrecting from previous life...")
            state = joblib.load(MODEL_PATH)
            self.history_buffer = state.get('buffer', [])
            self.scaler = state.get('scaler', self.scaler)
            self.is_trained = state.get('is_trained', False)
            
            # Instant Retrain
            if self.is_trained and len(self.history_buffer) > 0:
                 X = np.array(self.history_buffer)
                 self.model.fit(self.scaler.transform(X))
                 print(f"[Brain] Resurrection Complete. Memories restored: {len(self.history_buffer)}")
                 
        except Exception as e:
            print(f"[Brain] Resurrection Failed: {e}")

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

        # Vault Trigger A: Overflow
        if len(self.history_buffer) >= self.max_history:
             # Flush oldest 2000 items to Vault
             overflow = self.history_buffer[:2000]
             self.vault.flush(overflow)
             # Remove them from Active Memory
             self.history_buffer = self.history_buffer[2000:]

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

    def predict_hybrid(self, row):
        # The Hybrid Cascade Logic
        
        # 1. Tier 1: Deterministic Heuristic (Fastest)
        # Check simple rule-based traps
        if row[1] > 200 and row[0] < 1.0:
            return -1.0, -1.0, "SUSPICIOUS_COPY_TRAP"
            
        # 2. Tier 2: Isolation Forest (Fast ML)
        if not self.is_trained: return 0,0,[]
        
        try:
            X = np.array([row])
            X_scaled = self.scaler.transform(X)
            score = self.model.predict(X_scaled)[0]
            severity = self.model.score_samples(X_scaled)[0]
            
            # 3. Tier 3: Ambiguity Check (Auto-Audit)
            if -0.7 < score < -0.6:
                # Ambiguous! 
                # A. Fast Check: Vault (Tier 3)
                is_historic = self.vault.audit(row, tolerance=2.0)
                if is_historic:
                    score = 1; severity = 0.5 # Downgrade immediately
                else:
                    # B. Deep Check: Queue for Background (Tier 4)
                    # We pass through (don't block), but notify Ghost Thread
                    self.ghost.submit(row, score)
                
            return score, severity, []
        except:
            return 0,0,[]

# --- Helper for Printing ---
class SessionTracker:
    def __init__(self):
        self.stats = {} # Dynamic
        self.total = 0
        self.matches = 0
        self.alerts = 0
        self.last_print = time.time()
        self.start_time = time.time()

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

    def print_final_report(self):
        duration = time.time() - self.start_time
        print("\n\n" + "="*60)
        print(f"   SESSION ANALYSIS REPORT (Duration: {duration:.1f}s)")
        print("="*60)
        print(f"[-] Total Events Processed:  {self.total}")
        print(f"[-] Anomalies Detected:      {self.alerts}")
        print(f"[-] Knowledge Bank Matches:  {self.matches}")
        
        health = 100.0
        if self.total > 0:
            health = 100.0 - (self.alerts / self.total * 100.0)
        print(f"[-] Final System Health:     {health:.2f}%")
        
        print("\n[Activity Distribution]")
        sorted_stats = sorted(self.stats.items(), key=lambda item: item[1], reverse=True)
        for cat, count in sorted_stats:
            pct = (count / self.total) * 100 if self.total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"    {cat:<20} | {pct:5.1f}% | {bar}")
        print("="*60 + "\n")


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
                    
                    # 2. Check Anomaly (Using Hybrid Cascade)
                    score, severity, aux_data = brain.predict_hybrid([data['GPU_TIME_MS'], data['GPU_PACKET_COUNT']])
                    
                    known_labels = [] # Default initialization
                    
                    if aux_data == "SUSPICIOUS_COPY_TRAP":
                        # Specific handling for the Trap
                        score = -1
                        severity = -1.0
                    elif isinstance(aux_data, list):
                        known_labels = aux_data
                    
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
        print("\n[!] User Interrupted. Stopping Brain...")
        session.print_final_report()

if __name__ == "__main__":
    main()
