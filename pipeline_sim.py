"""
PipelineSimAI: CPU Pipeline Simulator with ML-Based Branch Prediction + Visualization

Features:
✔ Data hazards (stalling)
✔ Forwarding (optional)
✔ Control hazards (flush penalty)
✔ Branch prediction (static + ML-based)
✔ Performance metrics (CPI, stalls, flushes, mispredictions)
✔ Visualization with matplotlib (pipeline timing diagram)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# ------------------------------
# Instruction Class
# ------------------------------
class Instruction:
    def __init__(self, op, dest=None, src1=None, src2=None, imm=None):
        self.op = op
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
        self.imm = imm

    def __repr__(self):
        return f"{self.op} {self.dest or ''} {self.src1 or ''} {self.src2 or ''} {self.imm or ''}"


# ------------------------------
# Pipeline Stages
# ------------------------------
STAGES = ["IF", "ID", "EX", "MEM", "WB"]
STAGE_COLORS = {
    "IF": "lightblue",
    "ID": "orange",
    "EX": "lightgreen",
    "MEM": "violet",
    "WB": "pink"
}


# ------------------------------
# Data Hazard Detection
# ------------------------------
def has_data_hazard(curr_instr, prev_instr, prev_op, forwarding=True):
    if not curr_instr or not prev_instr:
        return False
    if forwarding:
        if prev_op == "LW":  # load-use hazard only
            return (curr_instr.src1 == prev_instr.dest) or (curr_instr.src2 == prev_instr.dest)
        return False
    else:
        return (curr_instr.src1 == prev_instr.dest) or (curr_instr.src2 == prev_instr.dest)


# ------------------------------
# ML Branch Predictor
# ------------------------------
class MLBranchPredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.trained = False

    def generate_training_data(self, n_samples=200):
        X, y = [], []
        for _ in range(n_samples):
            op = random.choice(["BEQ", "BNE"])
            src1_val = random.randint(0, 10)
            src2_val = random.randint(0, 10)
            last_outcome = random.choice([0, 1])  # previous taken or not

            if op == "BEQ":
                actual = 1 if src1_val == src2_val else 0
            else:
                actual = 1 if src1_val != src2_val else 0

            features = [0 if op == "BEQ" else 1, src1_val, src2_val, last_outcome]
            X.append(features)
            y.append(actual)
        return np.array(X), np.array(y)

    def train(self):
        X, y = self.generate_training_data()
        self.model.fit(X, y)
        self.trained = True

    def predict(self, instr, last_outcome=0):
        if not self.trained:
            self.train()
        op_enc = 0 if instr.op == "BEQ" else 1
        src1_val = random.randint(0, 10)  # mock register values
        src2_val = random.randint(0, 10)
        features = np.array([[op_enc, src1_val, src2_val, last_outcome]])
        return bool(self.model.predict(features)[0])


# ------------------------------
# Pipeline Simulation
# ------------------------------
def simulate_pipeline(instructions, forwarding=True, predictor=None):
    pipeline = [None]*5
    cycle = 0
    results = []
    instr_queue = instructions[:]

    stalls = 0
    flushes = 0
    mispredictions = 0
    executed_instr = len(instructions)

    last_outcome = 0

    while instr_queue or any(stage is not None for stage in pipeline):
        cycle += 1
        snapshot = {}
        stalled = False
        flush = False

        # Data hazard check
        if pipeline[1] and pipeline[2]:
            if has_data_hazard(pipeline[1], pipeline[2], pipeline[2].op, forwarding):
                stalled = True

        # Control hazard check
        if pipeline[2] and pipeline[2].op in ["BEQ", "BNE"]:
            if predictor is None:
                prediction = True  # static predictor (always taken)
            else:
                prediction = predictor.predict(pipeline[2], last_outcome)

            # For demo → assume branch is always TAKEN
            actual_taken = True
            last_outcome = 1 if actual_taken else 0

            if prediction != actual_taken:
                mispredictions += 1
                flush = True
            if actual_taken:
                flushes += 1

        # Pipeline update
        if stalled:
            stalls += 1
            pipeline[4] = pipeline[3]
            pipeline[3] = pipeline[2]
            pipeline[2] = None
        else:
            for i in range(4, 0, -1):
                pipeline[i] = pipeline[i-1]
            pipeline[0] = instr_queue.pop(0) if instr_queue else None

        if flush:
            pipeline[0] = None
            pipeline[1] = None

        for i, instr in enumerate(pipeline):
            snapshot[STAGES[i]] = instr.op if instr else "-"
        results.append((cycle, snapshot))

    CPI = cycle / executed_instr if executed_instr > 0 else 0
    metrics = {
        "cycles": cycle,
        "instructions": executed_instr,
        "stalls": stalls,
        "flushes": flushes,
        "mispredictions": mispredictions,
        "CPI": round(CPI, 2)
    }
    return results, metrics


# ------------------------------
# Print & Visualization
# ------------------------------
def print_results(results, metrics, title="Pipeline Simulation"):
    print(f"\n=== {title} ===")
    print("Cycle | " + " | ".join(STAGES))
    print("-"*50)
    for cycle, snapshot in results:
        row = [snapshot[s] for s in STAGES]
        print(f"{cycle:5} | " + " | ".join(row))
    print("\n--- Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k:15}: {v}")


def visualize_pipeline(results, title="Pipeline Diagram"):
    fig, ax = plt.subplots(figsize=(10, 6))

    instr_ids = {}
    instr_counter = 0

    for cycle, snapshot in results:
        for stage, instr in snapshot.items():
            if instr != "-":
                if instr not in instr_ids:
                    instr_ids[instr] = instr_counter
                    instr_counter += 1
                y = instr_ids[instr]
                ax.barh(y, 1, left=cycle-1, color=STAGE_COLORS[stage], edgecolor="black")

    ax.set_xlabel("Cycles")
    ax.set_ylabel("Instructions")
    ax.set_title(title)
    ax.set_yticks(list(instr_ids.values()))
    ax.set_yticklabels(list(instr_ids.keys()))
    plt.tight_layout()
    plt.show()


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    instr_list = [
        Instruction("ADD", "R1", "R2", "R3"),
        Instruction("SUB", "R4", "R1", "R5"),
        Instruction("LW", "R6", "R4", imm=0),
        Instruction("ADD", "R7", "R6", "R1"),
        Instruction("BEQ", src1="R1", src2="R2", imm=8),
        Instruction("SW", "R7", "R2", imm=4)
    ]

    # Run with static predictor
    results, metrics = simulate_pipeline(instr_list, forwarding=True, predictor=None)
    print_results(results, metrics, title="Static Predictor (Always Taken)")
    visualize_pipeline(results, title="Pipeline (Static Predictor)")

    # Run with ML predictor
    ml_predictor = MLBranchPredictor()
    results, metrics = simulate_pipeline(instr_list, forwarding=True, predictor=ml_predictor)
    print_results(results, metrics, title="ML-Based Branch Predictor")
    visualize_pipeline(results, title="Pipeline (ML Branch Predictor)")
