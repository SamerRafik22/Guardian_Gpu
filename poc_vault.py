
import numpy as np
import time
import os
from guardian_brain import GuardianBrain, GuardianVault

# Cleanup previous tests
if os.path.exists("guardian_vault.jsonl.gz"):
    os.remove("guardian_vault.jsonl.gz")

print("=== Guardian Vault PoC Generator ===")

print("\n[Step 1] Initializing Brain...")
brain = GuardianBrain()
brain.is_trained = True 
brain.scaler.fit([[0,0], [100,500]]) # Dummy fit

# Synthetic Data Generation
print("\n[Step 2] Generating 'Ancient' History (e.g. Last Month's Gameplay)...")
# Let's say user played a game with specific metrics: 42ms Time, 88 Packets
ancient_game_trace = [[42.0, 88.0] for _ in range(2500)]

# Inject into Brain buffer
brain.history_buffer.extend(ancient_game_trace)
print(f"Brain Buffer Size: {len(brain.history_buffer)}")

print("\n[Step 3] Triggering Vault Flush (Overflow)...")
# Force overflow logic manually or by adding more data
# Let's add dummy data to push it over 10,000 limit
dummy_fill = [[10.0, 20.0] for _ in range(8000)]
brain.history_buffer.extend(dummy_fill)

print(f"Buffer before flush: {len(brain.history_buffer)}")
# Call predict on dummy data to trigger the overflow check logic inside predict
# Or simpler: manually invoke the flush logic for this PoC to be deterministic
overflow = brain.history_buffer[:2500] 
brain.vault.flush(overflow)
brain.history_buffer = brain.history_buffer[2500:]

print(f"Buffer after flush: {len(brain.history_buffer)}")
if os.path.exists("guardian_vault.jsonl.gz"):
    print(">> SUCCESS: Vault File Created on Disk!")
else:
    print(">> FAIL: Vault File Missing!")

print("\n[Step 4] Simulating 'Ambiguous' Today Event...")
# New event happens today that looks EXACTLY like that ancient game
# It's ambiguous because it's distinct from the "Dummy Fill" (Recent Memory)
today_event = [42.0, 88.0] 

# Manually trigger Audit for PoC
print(f"Analyzing Event: {today_event}")
is_found = brain.vault.audit(today_event, tolerance=0.1)

if is_found:
    print(f">> SUCCESS: Cloud Audit confirmed this pattern exists in Vault!")
    print("   Result: False Alarm prevented.")
else:
    print(">> FAIL: Pattern not found in Vault.")

print("\n=== PoC Complete ===")
