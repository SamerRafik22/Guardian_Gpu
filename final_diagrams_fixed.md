# Guardian GPU - Final Diagrams (Aligned to Code)
Date: 2026-02-06

Notes
- Aligned with: monitor_main.cpp, gpu_mon.h/cpp, etw_monitor.h/cpp, csv_logger.h, guardian_brain.py
- Focused on the active ETW + NVML + Python Brain pipeline
- Legacy CUPTI injection in legacy/ is not represented here

---

## Figure 1.1: System Overview (Detailed, Code-Aligned)
```mermaid
flowchart TB
  subgraph Host["Windows Host"]
    App["GPU Workload (User App)"]
    Dxg["DxgKrnl ETW Provider"]
    GPU["GPU Hardware"]
    Driver["NVIDIA Driver (NVML)"]
  end

  subgraph Monitor["GuardianMonitor (C++)"]
    Loop1["1s Main Loop"]
    Etw["EtwMonitor"]
    EtwMap["s_gpuUsageMap\n(busyTimeMs, packetCount)"]
    Nvml["GpuMonitor (NVML)"]
    Dev["Device Metrics\n(power, temp, fan, clocks, VRAM)"]
    Csv["CsvLogger"]
    CppConsole["C++ Console Table"]
  end

  subgraph Brain["GuardianBrain (Python)"]
    Streamer["LogStreamer (tail CSV)"]
    Parse["parse_line()"]
    Classify["classify_activity()"]
    Predict["predict_hybrid()"]
    Heuristic["Heuristic Trap\n(packet>200 & time<1)"]
    Iso["IsolationForest + StandardScaler"]
    History["history_buffer (max 10k)\nretrain every 50 events"]
    Vault["GuardianVault\n(jsonl.gz append)"]
    Ghost["BackgroundAnalyzer\n(ghost thread)"]
    KB["KnowledgeBank\n(loaded JSON)"]
    PyConsole["Python Console Alerts/Summary"]
    BrainState["brain_state.pkg\n(save/load)"]
  end

  App --> Dxg
  Dxg -. "ETW events" .-> Etw
  GPU -. "telemetry" .-> Driver
  Driver --> Nvml

  Loop1 --> Etw --> EtwMap
  Loop1 --> Nvml --> Dev
  EtwMap --> Csv
  Dev --> Csv
  Loop1 --> CppConsole

  Csv --> Log["gpu_log_YYYYMMDD_HHMMSS.csv"]
  Log --> Streamer --> Parse --> Classify --> Predict

  Predict --> Heuristic
  Predict --> Iso
  Predict --> History
  History --> Vault
  Predict --> PyConsole

  Predict -. "ambiguous (-0.7 < score < -0.6)" .-> Ghost
  Ghost --> KB
  Ghost --> Vault

  BrainState --> Predict
```

---

## Figure 3.1: Use Case Diagram (Aligned to Code)
```mermaid
flowchart LR
  Operator["System Operator"]
  Workload["GPU Workload"]

  subgraph System["Guardian GPU System"]
    UC_StartMon(("Start GuardianMonitor"))
    UC_StartBrain(("Start GuardianBrain"))
    UC_Stop(("Stop / Ctrl+C"))
    UC_View(("View console output"))

    UC_ETW(("Collect ETW GPU Events"))
    UC_NVML(("Collect NVML Telemetry"))
    UC_CSV(("Write CSV Log"))

    UC_Stream(("Stream Latest CSV"))
    UC_Classify(("Classify Activity"))
    UC_Detect(("Detect Anomalies"))
    UC_Summary(("Print Alerts/Summary"))
    UC_Save(("Save Brain State"))
    UC_Vault(("Flush History to Vault"))
  end

  Operator --> UC_StartMon
  Operator --> UC_StartBrain
  Operator --> UC_Stop
  Operator --> UC_View

  Workload -.-> UC_ETW
  Workload -.-> UC_NVML

  UC_StartMon -.->|"include"| UC_ETW
  UC_StartMon -.->|"include"| UC_NVML
  UC_StartMon -.->|"include"| UC_CSV

  UC_StartBrain -.->|"include"| UC_Stream
  UC_StartBrain -.->|"include"| UC_Classify
  UC_StartBrain -.->|"include"| UC_Detect
  UC_StartBrain -.->|"include"| UC_Summary

  UC_CSV --> UC_Stream
  UC_Detect -.->|"extend (overflow)"| UC_Vault
  UC_Stop -.->|"include"| UC_Save
```

---

## Figure 4.1: Blackbox (Inputs and Outputs)
```mermaid
flowchart LR
  subgraph Inputs
    ETW["ETW GPU events (DxgKrnl)"]
    NVML["NVML telemetry (power/VRAM/etc)"]
  end

  System["Guardian GPU System"]

  subgraph Outputs
    CSV["CSV logs (gpu_log_*.csv)"]
    Console["Console metrics + alerts"]
  end

  ETW --> System
  NVML --> System
  System --> CSV
  System --> Console
```

---

## Figure 4.2: MVC Architecture (C++ Monitor + Python Brain)
```mermaid
flowchart TD
  subgraph Controller["Controller"]
    C_CPP["monitor_main.cpp (main loop)"]
    C_PY["GuardianBrain (analysis loop)"]
    C_GHOST["BackgroundAnalyzer (thread)"]
  end

  subgraph Model["Model"]
    M_ETW["EtwMonitor (ETW events)"]
    M_NVML["GpuMonitor (NVML telemetry)"]
    M_CSV["CsvLogger + gpu_log_*.csv"]
    M_IF["IsolationForest + StandardScaler"]
    M_KB["KnowledgeBank (JSON)"]
    M_VAULT["GuardianVault (JSONL.gz)"]
  end

  subgraph View["View"]
    V_CPP["C++ Console Table"]
    V_PY["Python Console Alerts/Summary"]
  end

  C_CPP --> M_ETW
  C_CPP --> M_NVML
  C_CPP --> M_CSV
  C_CPP --> V_CPP

  C_PY --> M_CSV
  C_PY --> M_IF
  C_PY --> M_KB
  C_PY --> M_VAULT
  C_PY --> V_PY

  C_GHOST --> M_VAULT
  C_GHOST --> M_KB
```

---

## Figure 4.3: UML Component Diagram
```mermaid
flowchart LR
  %% UML component-style diagram (clean labels)

  subgraph Cpp["C++ Monitor Components"]
    GM["GuardianMonitor.exe"]
    ETW["EtwMonitor"]
    NVML["GpuMonitor"]
    CSV["CsvLogger"]
  end

  subgraph Py["Python Brain Components"]
    GB["GuardianBrain.py"]
    LS["LogStreamer"]
    IF["IsolationForest"]
    KB["KnowledgeBank"]
    GV["GuardianVault"]
    BG["BackgroundAnalyzer"]
  end

  CSVLOG["gpu_log_*.csv (artifact)"]
  BRAIN["brain_state.pkg (artifact)"]
  KBFILE["knowledge_bank.json (artifact)"]
  VAULT["guardian_vault.jsonl.gz (artifact)"]

  GM --> ETW
  GM --> NVML
  GM --> CSV
  CSV -->|writes| CSVLOG

  GB --> LS
  GB --> IF
  GB --> KB
  GB --> GV
  GB --> BG

  LS -->|reads| CSVLOG
  GB -->|save/load| BRAIN
  KB -->|loads| KBFILE
  GV -->|read/write| VAULT

  BG -->|audit/flush| GV
  BG -->|append in-memory| KB
```

---

## Figure 4.4: Class Diagram (Key Types)
```mermaid
classDiagram
  class GpuMonitor {
    +Initialize()
    +Shutdown()
    +CaptureProcessSnapshots(device)
    +CaptureDeviceMetrics(device)
  }

  class EtwMonitor {
    +StartSession()
    +StopSession()
    +PopProcessUsage()
  }

  class CsvLogger {
    +LogRow(timestamp, pid, name, vram, power, gpuTimeMs, packetCount, netTx, netRx)
  }

  class GpuProcessSnapshot {
    +uint32 pid
    +string processName
    +uint64 vramUsedBytes
    +uint utilizationGpu
    +uint32 kernelCount
    +double durationMeanNs
    +double durationStdDevNs
    +double occupancy
  }

  class GpuDeviceMetrics {
    +uint powerUsage
    +uint temperature
    +uint fanSpeed
    +uint coreClock
    +uint memClock
    +uint pcieRx
    +uint pcieTx
  }

  class ProcessStats {
    +double busyTimeMs
    +uint32 packetCount
    +uint64 netTxBytes
    +uint64 netRxBytes
  }

  class GuardianBrain {
    +train_initial(path)
    +predict(line_data)
    +predict_hybrid(row)
    +retrain_dynamic()
    +save_state()
    +load_state()
  }

  class LogStreamer {
    +get_latest_log()
    +stream_lines()
  }

  class KnowledgeBank {
    +load()
    +is_known(vector)
  }

  class GuardianVault {
    +audit(vector)
    +flush(chunk)
  }

  class BackgroundAnalyzer {
    +submit(data_row, score)
    +run()
  }

  class SessionTracker {
    +update(category)
    +print_summary()
    +print_final_report()
  }

  EtwMonitor --> ProcessStats
  GpuMonitor --> GpuProcessSnapshot
  GpuMonitor --> GpuDeviceMetrics

  GuardianBrain *-- LogStreamer
  GuardianBrain *-- KnowledgeBank
  GuardianBrain *-- GuardianVault
  GuardianBrain *-- BackgroundAnalyzer
  GuardianBrain ..> SessionTracker
```

---

## Figure 4.5: Operational Flow (Exact to Current Code)
```mermaid
flowchart TD
  Start([Start])

  %% C++ Monitor
  subgraph CppLoop["C++ Monitor Loop (1s interval)"]
    CppInit["NVML init + ETW session"]
    CppTick["Capture NVML + ETW"]
    CppLog["Write CSV row"]
    CppTick --> CppLog --> CppTick
  end

  %% Python Brain
  subgraph PyLoop["Python Brain Loop"]
    PyInit["load_state + train_initial"]
    PyRead["LogStreamer.stream_lines"]
    PyParse["parse_line"]
    PyClassify["classify_activity"]
    PyPredict["predict_hybrid([GPU_TIME_MS, GPU_PACKET_COUNT])"]

    PyKB["known_labels = [] (predict returns none)"]
    PyRule1{"category == SUSPICIOUS_COPY?"}
    PyForceAnom["score=-1; severity=-0.9999"]
    PyScore{"score == -1 ?"}
    PyAlert["Print anomaly details"]
    PySummary["print_summary (every ~2s)"]

    PyRead --> PyParse --> PyClassify --> PyPredict --> PyKB --> PyRule1
    PyRule1 -->|yes| PyForceAnom --> PyScore
    PyRule1 -->|no| PyScore
    PyScore -->|yes| PyAlert --> PySummary --> PyRead
    PyScore -->|no| PySummary --> PyRead
  end

  Start --> CppInit --> CppTick
  Start --> PyInit --> PyRead

  %% predict_hybrid internal logic
  PyPredict --> H1{"packet>200 and time<1ms?"}
  H1 -->|yes| RetTrap["return score=-1 (trap)"]
  H1 -->|no| IF["IsolationForest.predict()"]
  IF --> Gray{"-0.7 < score < -0.6?"}
  Gray -->|yes| VaultAudit["GuardianVault.audit(row)"]
  Gray -->|no| RetScore["return score"]
  VaultAudit -->|found| RetSafe["score=1 (safe)"]
  VaultAudit -->|not found| Ghost["ghost.submit(row, score)"]
  Ghost --> RetScore
```

---

## Figure 4.6: ERD (Full, Code-Aligned)
```mermaid
erDiagram
  %% File-based storage (no real foreign keys). Relations are conceptual.

  CSV_LOG_ROW {
    string TIMESTAMP
    int PID
    string NAME
    int MEM_MB
    float PWR_W
    float GPU_TIME_MS
    int GPU_PACKET_COUNT
    int NET_TX
    int NET_RX
  }

  VAULT_VECTOR {
    float GPU_TIME_MS
    int GPU_PACKET_COUNT
  }

  KNOWLEDGE_SIGNATURE {
    string LABEL
    float VECTOR_GPU_TIME_MS
    int VECTOR_GPU_PACKET_COUNT
  }

  BRAIN_STATE {
    bool IS_TRAINED
    int HISTORY_BUFFER_LEN
  }

  CSV_LOG_ROW }o--o{ VAULT_VECTOR : "overflow -> vault.flush"
  VAULT_VECTOR }o--o{ KNOWLEDGE_SIGNATURE : "ghost may whitelist"
  BRAIN_STATE }o--o{ VAULT_VECTOR : "history_buffer stores vectors"
```
