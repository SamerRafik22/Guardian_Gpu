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
    Api["Web API Gateway\n(JSON Interface)"]
  end

  subgraph WebUI["Web Dashboard (New UI)"]
    Dash["Dashboard UI\n(Graphs/Stats)"]
    Proc["Process Manager\n(Table/Controls)"]
    Toast["Toast Alerts\n(Notifications)"]
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
  Predict --> Api
  Predict --> PyConsole

  Predict -. "ambiguous (-0.7 < score < -0.6)" .-> Ghost
  Ghost --> KB
  Ghost --> Vault

  BrainState --> Predict
  
  Api --> Dash
  Api --> Proc
  Api --> Toast
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

## Figure 4.2: MVC Architecture (C++ Monitor + Python Brain + Web UI)
```mermaid
flowchart TD
  subgraph Controller["Controller"]
    C_CPP["monitor_main.cpp (main loop)"]
    C_PY["GuardianBrain (analysis loop)"]
    C_GHOST["BackgroundAnalyzer (thread)"]
    C_WEB["WebController / API Gateway"]
  end

  subgraph Model["Model"]
    M_ETW["EtwMonitor (ETW events)"]
    M_NVML["GpuMonitor (NVML telemetry)"]
    M_CSV["CsvLogger + gpu_log_*.csv"]
    M_IF["IsolationForest + StandardScaler"]
    M_KB["KnowledgeBank (JSON)"]
    M_VAULT["GuardianVault (JSONL.gz)"]
  end

  subgraph View["View (New Web UI)"]
    V_CPP["C++ Console Table"]
    V_PY["Python Console Alerts/Summary"]
    V_DASH["Dashboard UI (Charts/Metrics)"]
    V_PROC["Process Manager (List/Actions)"]
    V_TOAST["Toast Notifications (Alerts)"]
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
  C_PY --> C_WEB

  C_GHOST --> M_VAULT
  C_GHOST --> M_KB
  
  C_WEB --> V_DASH
  C_WEB --> V_PROC
  C_WEB --> V_TOAST
  
  V_PROC -.->|"terminate process"| C_WEB
```

---

## Figure 4.3: UML Component Diagram (Formal)
```mermaid
classDiagram
  %% UML-style component diagram (Mermaid-renderable)

  class DxgKrnlProvider {
    <<external>>
    +emitEtwEvents()
  }
  class NvmlProvider {
    <<external>>
    +serveTelemetry()
  }

  class GuardianMonitorExe {
    <<component>>
    +mainLoop()
  }
  class EtwMonitor {
    <<component>>
    +PopProcessUsage()
  }
  class GpuMonitor {
    <<component>>
    +CaptureDeviceMetrics()
  }
  class CsvLogger {
    <<component>>
    +LogRow()
  }

  class GuardianBrain {
    <<component>>
    +predict_hybrid()
  }
  class LogStreamer {
    <<component>>
    +stream_lines()
  }
  class IsolationForest {
    <<component>>
    +predict()
  }
  class KnowledgeBank {
    <<component>>
    +is_known()
  }
  class GuardianVault {
    <<component>>
    +audit()
  }
  class BackgroundAnalyzer {
    <<component>>
    +submit()
  }
  class SessionTracker {
    <<component>>
    +print_summary()
  }

  class WebController {
    <<component>>
    +status: planned
    +serve_api()
  }
  class WebDashboard {
    <<component>>
    +status: planned
    +render_metrics()
  }
  class ProcessManagerUI {
    <<component>>
    +status: planned
    +list_processes()
  }
  class ToastNotifier {
    <<component>>
    +status: planned
    +push_alert()
  }

  class IEtwUsage {
    <<interface>>
    +PopProcessUsage()
  }
  class INvmlSnapshot {
    <<interface>>
    +CaptureDeviceMetrics()
  }
  class ILogWrite {
    <<interface>>
    +LogRow()
  }
  class ILogRead {
    <<interface>>
    +stream_lines()
  }
  class IAnomalyInference {
    <<interface>>
    +predict()
  }
  class IKBLookup {
    <<interface>>
    +is_known()
  }
  class IVaultAudit {
    <<interface>>
    +audit()
  }
  class IWebUiFeed {
    <<interface>>
    +get_status()
  }
  class IProcessActions {
    <<interface>>
    +terminate_process()
  }

  class CsvLog {
    <<artifact>>
    +gpu_log_csv
  }
  class BrainState {
    <<artifact>>
    +brain_state_pkg
  }
  class KBFile {
    <<artifact>>
    +knowledge_bank_json
  }
  class VaultFile {
    <<artifact>>
    +guardian_vault_jsonl_gz
  }
  class UiHtml {
    <<artifact>>
    +ui_html
  }
  class DashboardMockup {
    <<artifact>>
    +dashboard_mockup_png
  }

  DxgKrnlProvider --> EtwMonitor : ETW events
  NvmlProvider --> GpuMonitor : telemetry

  EtwMonitor ..|> IEtwUsage
  GpuMonitor ..|> INvmlSnapshot
  CsvLogger ..|> ILogWrite
  LogStreamer ..|> ILogRead
  IsolationForest ..|> IAnomalyInference
  KnowledgeBank ..|> IKBLookup
  GuardianVault ..|> IVaultAudit
  WebController ..|> IWebUiFeed
  WebController ..|> IProcessActions

  GuardianMonitorExe --> IEtwUsage : uses
  GuardianMonitorExe --> INvmlSnapshot : uses
  GuardianMonitorExe --> ILogWrite : uses

  CsvLogger --> CsvLog : writes
  LogStreamer --> CsvLog : reads

  GuardianBrain --> ILogRead : uses
  GuardianBrain --> IAnomalyInference : uses
  GuardianBrain --> IKBLookup : requires
  GuardianBrain --> IVaultAudit : requires
  BackgroundAnalyzer --> IKBLookup : requires
  BackgroundAnalyzer --> IVaultAudit : requires

  GuardianBrain *-- KnowledgeBank : owns
  GuardianBrain *-- GuardianVault : owns
  GuardianBrain *-- BackgroundAnalyzer : owns
  GuardianBrain *-- SessionTracker : owns
  GuardianBrain --> BrainState : save_load
  KnowledgeBank --> KBFile : loads
  GuardianVault --> VaultFile : read_write
  BackgroundAnalyzer --> GuardianVault : audit_flush
  BackgroundAnalyzer --> KnowledgeBank : append_in_memory

  GuardianBrain --> WebController : feeds_status
  WebController --> WebDashboard : serves
  WebController --> ProcessManagerUI : serves
  WebController --> ToastNotifier : pushes_alerts
  ProcessManagerUI --> IProcessActions : requests_action
  WebController --> UiHtml : serves
  WebDashboard --> DashboardMockup : references
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

  %% Nested struct in code: EtwMonitor::ProcessStats
  class EtwProcessStats {
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

  EtwMonitor --> EtwProcessStats
  GpuMonitor --> GpuProcessSnapshot
  GpuMonitor --> GpuDeviceMetrics

  GuardianBrain *-- LogStreamer
  GuardianBrain *-- KnowledgeBank
  GuardianBrain *-- GuardianVault
  GuardianBrain *-- BackgroundAnalyzer
  GuardianBrain ..> SessionTracker
```

---

## Figure 4.5: UML Activity Diagram (Operational Flow)
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
    int log_id PK "Primary Key (Auto-inc)"
    string timestamp
    int pid
    string name
    int mem_mb
    float pwr_w
    float gpu_time_ms
    int gpu_packet_count
    int net_tx
    int net_rx
  }

  VAULT_VECTOR {
    int vector_id PK "Primary Key"
    int log_ref_id FK "Foreign Key (links to CSV_LOG_ROW)"
    int signature_id FK "Foreign Key (links to KNOWLEDGE_SIGNATURE.signature_id)"
    float gpu_time_ms
    int gpu_packet_count
    string archived_at
  }

  KNOWLEDGE_SIGNATURE {
    int signature_id PK "Primary Key"
    string label "Activity Name"
    float center_gpu_time_ms
    int center_packet_count
    float threshold_radius
  }

  BRAIN_STATE {
    int sess_id PK "Primary Key (Session ID)"
    bool is_trained
    int history_buffer_len
    string last_updated
  }

  CSV_LOG_ROW ||--o{ VAULT_VECTOR : "archives_as (1:N)"
  KNOWLEDGE_SIGNATURE ||--o{ VAULT_VECTOR : "classifies (1:N)"
  BRAIN_STATE ||--o{ CSV_LOG_ROW : "monitors (1:N)"
```

---

## Figure 4.7: UML Sequence Diagram (End-to-End Flow)
```mermaid
sequenceDiagram
  participant Monitor as GuardianMonitor.exe
  participant ETW as EtwMonitor
  participant NVML as GpuMonitor
  participant CSV as CsvLogger
  participant Log as gpu_log_*.csv
  participant Stream as LogStreamer
  participant Brain as GuardianBrain
  participant IF as IsolationForest
  participant Ghost as BackgroundAnalyzer
  participant Vault as GuardianVault
  participant KB as KnowledgeBank

  loop Every 1s
    Monitor->>ETW: PopProcessUsage()
    Monitor->>NVML: CaptureDeviceMetrics()
    Monitor->>CSV: LogRow(...)
    CSV->>Log: append row
  end

  loop Continuous
    Stream->>Log: read new lines
    Stream-->>Brain: parsed row
    Brain->>IF: predict()
    alt Heuristic Trap (count>200 & time<1)
      Brain-->>Brain: score = -1
    else Normal path
      IF-->>Brain: score (1 or -1)
    end
    alt score == -1
      Brain-->>Brain: print alert
    else score == 1
      Brain-->>Brain: update summary
    end
    opt Ambiguous range (code path)
      Brain->>Vault: audit(row)
      Brain->>Ghost: submit(row, score)
      Ghost->>KB: append in-memory (if safe)
    end
  end
```


