# Writing for Machines — Experiment Code

Replication code for: *"Writing for Machines: How Symbolic Expression 
Can Reduce Energy Consumption in Large Language Models"*

## Setup
pip install -r requirements.txt

## Experiments

### Claim B — Noise experiment (Table 1)
python experiment.py

### Claim A — Tokenizer-friendly symbols (Table 2 + Figure 2)
python claim_a.py

### Figure 2 regeneration
python fig2_redesign.py

## Results
Raw aggregated results are saved in results.json and results_claim_a.json
