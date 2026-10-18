## Introduction
InteractiveGNNExplainer, a visual analytics framework to enhance GNN explainability.  
Paper published in 29th International Conference Information Visualisation 2025 (https://ieeexplore.ieee.org/abstract/document/11216840)

## Offline Phase
To start first run the following codes:
1. preprocess_dataset.py
2. train_models.py
3. precompute_stability.py 
4. precompute_link_stability.py

## Online Phase
Run the dash application using app.py. By default it will start at http://localhost:8050/

The server binds to 127.0.0.1 with debug mode on. To serve on your network, set
`GNN_VIS_HOST=0.0.0.0` and turn debug off with `GNN_VIS_DEBUG=0` (the debugger allows code execution).

Quick check after the offline phase: `python test_smoke.py`


For better performance GPU should be utilized
