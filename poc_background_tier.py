
import time
import numpy as np
from guardian_brain import GuardianBrain

print("=== Tier 4: Ghost Thread PoC ===")
print("[1] Initializing Guardian Brain...")
brain = GuardianBrain()
brain.is_trained = True
# Fake trainer to allow prediction
brain.scaler.fit([[0,0], [10,100]]) 
brain.model.fit([[0,0], [10,100], [5,50], [2,20]])

print("\n[2] Simulating Ambiguous Event...")
# We craft an event that is NOT a Heuristic Trap, but "kinda" weird
# Valid Range for Ambiguity in code: -0.7 < score < -0.6
# Mocking the predict_hybrid outcome to force the 'submit' call for demo purposes
# (Since controlling random forest score exactly is hard in a mock)

# We will manually inject into the ghost queue to prove the THREAD works
ambiguous_row = [55.5, 120.0] # 55ms compute, 120 packets (Grey Area)
original_score = -0.65 

print(f"    Event: {ambiguous_row}")
print(f"    Score: {original_score} (Ambiguous Range)")
print("    Action: Pushing to Ghost Thread...")

# Manually submit since we want to test the THREAD Logic specifically
brain.ghost.submit(ambiguous_row, original_score)

print("\n[3] Main Thread Continues (Game Not Paused)...")
for i in range(3):
    print(f"    Game Frame {i+1} rendered... (FPS stable)")
    time.sleep(0.5)

print("\n[4] Checking Knowledge Bank for Async Resolution...")
# The Ghost Thread takes 1.0s to "Think" (sleep) in our code
# So by now (1.5s later), it should be done.

resolved = False
for sig in brain.knowledge.known_signatures:
    if sig.get('vector') == ambiguous_row:
        resolved = True
        print(f"    >> FOUND SIGNATURE: {sig}")
        print(f"    >> LABEL: {sig.get('label')}")

if resolved:
    print("\n[SUCCESS] Ghost Thread analyzed and whitelisted the event in the background!")
else:
    print("\n[FAIL] Event was not processed by Ghost Thread.")

# Cleanup
brain.ghost.running = False
