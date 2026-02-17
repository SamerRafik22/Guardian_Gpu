# PDF vs Code Gaps (Guardian GPU)
Date: 2026-02-06

Purpose
- This file lists concrete mismatches between the PDF and the current codebase.
- Use it as a fix list for the document.

---

## A. Core Architecture & Runtime
1. PDF says "kernel-level HIDS" and "kernel-level telemetry collection".
2. Code reality: user-mode ETW consumer (no kernel driver).
3. Fix: say "user-mode ETW consumer of DxgKrnl".

4. PDF says 10Hz / 100ms polling.
5. Code reality: 1Hz loop (`Sleep(1000)` in `monitor_main.cpp`).
6. Fix: update to 1Hz.

7. PDF says Named Pipe IPC between C++ and Python.
8. Code reality: file-based CSV only.
9. Fix: remove pipe mention; state file-based streaming.

10. PDF says `gpu-log.csv` or `GPU_Log_YYYYMMDD.csv` naming.
11. Code reality: `gpu_log_YYYYMMDD_HHMMSS.csv` per run.
12. Fix: update filename format.

13. PDF says GUI (Electron/React), toast notifications.
14. Code reality: console only (C++ + Python).
15. Fix: mark GUI as future work or remove.

16. PDF says automated mitigation (kill/suspend) at certain scores.
17. Code reality: no kill/suspend logic.
18. Fix: remove or mark as future work.

---

## B. Detection Logic & Thresholds
1. PDF uses anomaly thresholds like -0.5 / -0.8.
2. Code uses `IsolationForest.predict()` which returns only -1 or 1.
3. Fix: describe decision as `score == -1` (anomaly) or adjust code.

4. PDF says Knowledge Bank whitelist is checked for every event (fast path).
5. Code reality: KnowledgeBank is not checked in `predict_hybrid`; used only by Ghost thread.
6. Fix: either update code or remove fast-path claim.

7. PDF says LOF is used for deep analysis.
8. Code reality: LOF imported but not used (mock logic in Ghost thread).
9. Fix: remove LOF claim or implement it.

10. PDF says continuous weekly retraining from Vault.
11. Code reality: no weekly retraining; only in-memory retrain every 50 events.
12. Fix: update description to match current retrain behavior.

---

## C. Features & Data Model
1. PDF says feature vector is 6D (GPU_TIME_MS, PACKET_COUNT, VRAM_MB, KERNEL_COUNTS, GPU_UTILIZATION, PID).
2. Code reality: 2D only (GPU_TIME_MS, GPU_PACKET_COUNT).
3. Fix: update feature list or add features in code.

4. PDF references KERNEL_COUNTS and GPU_UTILIZATION per process.
5. Code reality: KERNEL_COUNTS not computed; `utilizationGpu` stays 0.
6. Fix: remove or implement.

7. PDF says timestamps and detailed entities (EventId, VaultEntryId, etc.).
8. Code reality: JSONL vault stores raw vectors only; no IDs.
9. Fix: remove those entity fields or implement schema.

---

## D. Classes / Components Mismatch
1. PDF mentions classes not in code: UnifiedLogger, DecisionOrchestrator, FeatureExtractor, NvmlWrapper.
2. Code uses: CsvLogger, EtwMonitor, GpuMonitor, GuardianBrain, LogStreamer.
3. Fix: rename in document or add missing classes in code.

4. PDF lists EtwMonitor methods like `CaptureGPUTimestamp()` / `CaptureContextSwitch()`.
5. Code reality: `StartSession()` and `PopProcessUsage()` only.
6. Fix: update method list.

---

## E. Storage & Persistence
1. PDF claims Vault rotation at 100MB, retention (7 days hot / 30 days cold).
2. Code reality: no rotation/retention logic.
3. Fix: remove or implement.

4. PDF shows Knowledge Bank schema with `version`, `signatures`, `radius`, etc.
5. Code reality: expects `vector` + `label` (and current JSON is not aligned).
6. Fix: unify schema (doc + code + actual JSON file).

---

## F. UI / CLI Text Claims
1. PDF says GUI popup with Kill/Suspend/Leave actions.
2. Code reality: not implemented.
3. Fix: mark as future work or remove.

4. PDF claims nanosecond packet timing in console.
5. Code reality: GPU_TIME_MS in ms.
6. Fix: update units.

---

## G. Summary (What To Change In PDF)
1. Replace kernel-level claims with user-mode ETW consumer.
2. Remove Named Pipe / GUI / mitigation statements unless implemented.
3. Align feature vector to 2D or implement 6D in code.
4. Align classes, methods, and filenames to actual code.
5. Remove weekly retraining, vault rotation, LOF unless implemented.
6. Correct polling rate (1Hz) and units (ms).

