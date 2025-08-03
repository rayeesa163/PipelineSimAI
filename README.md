# PipelineSimAI
# 🚀 PipelineSimAI: CPU Pipeline Simulator with AI-Based Branch Prediction

## 📌 Overview
PipelineSimAI is a **CPU pipeline simulator** that models a 5-stage instruction pipeline 
(**IF, ID, EX, MEM, WB**) and handles hazards (stalling, forwarding, flushing).  
It also integrates an **AI-based branch predictor** (Decision Tree) to improve performance.  
The project includes a **Matplotlib visualization** of pipeline execution.

---

## 🎯 Features
- ✅ 5-Stage CPU pipeline simulation (IF, ID, EX, MEM, WB)
- ✅ Hazard detection and resolution (stalling, forwarding, flushing)
- ✅ Branch prediction:
  - Static predictors (Always Taken, Always Not Taken)
  - AI-based predictor (Decision Tree Classifier)
- ✅ Performance metrics: Cycles, Stalls, Flushes, Mispredictions, CPI
- ✅ Visualization with Matplotlib (pipeline timing diagram)

---

## 🖥️ Demo:https://www.loom.com/share/4f648b43654d4297a856d019b8483b2a?sid=c1c6f5a4-eed4-4930-baf1-ace6929246bf
### Pipeline Output Table
Instruction1 | IF | ID | EX | MEM | WB
Instruction2 | | IF | ID | EX | MEM | WB
...

### Performance Metrics
Total Cycles: 20
Stalls: 3
Flushes: 1
Mispredictions: 1
CPI: 1.25

### Visualization
![Pipeline Visualization]<img width="1241" height="737" alt="image" src="https://github.com/user-attachments/assets/8ae8905c-28ab-481e-b38b-ac959b658a73" />


---

## 📊 System Architecture
- Instruction class to model instructions
- Hazard detection logic
- AI branch predictor using Decision Tree
- Pipeline simulation loop
- Matplotlib visualization
  

---

## 📦 Installation & Usage
```bash
  # Clone repository
  git clone https://github.com/yourusername/PipelineSimAI.git
  cd PipelineSimAI
  
  # Install dependencies
  pip install -r requirements.txt
  
  # Run simulator
python pipeline_simulator.py

