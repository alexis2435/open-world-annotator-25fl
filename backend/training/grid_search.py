import itertools
from inference import run_single_inference

# Define grid search space
param_grid = {
    "confidence_threshold": [0.3, 0.5, 0.7],
    "top_k": [3, 5, 10],
    "merge_iou_threshold": [0.6, 0.75, 0.9]
}

# Generate all combinations
param_combinations = list(itertools.product(*param_grid.values()))
param_dicts = [dict(zip(param_grid.keys(), values)) for values in param_combinations]

# Run each config
for i, params in enumerate(param_dicts):
    print(f"\n=== Grid Search Run {i + 1}/{len(param_dicts)} ===")
    print("Params:", params)
    run_single_inference(**params)
