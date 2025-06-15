import torch
from torch_geometric.datasets import Planetoid
import os
import time
import traceback

DATA_DIR = 'data'
# Subfolder within DATA_DIR to store the processed .pt files.
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

DATASETS_TO_PREPROCESS = ['Cora', 'CiteSeer']

# --- Pre-processing Function ---
def preprocess_and_save(dataset_name, root_data_dir, processed_data_output_dir):
  
    print(f"Processing '{dataset_name}'...")
    start_time = time.time()

    try:
        # --- Step 1: Loading the dataset  ---
        print(f"  Loading '{dataset_name}' with PyG's Planetoid loader...")
        
        if dataset_name not in ['Cora', 'CiteSeer']:
            print(f" ERROR: Dataset '{dataset_name}' is not supported by this script.")
            return False
            
        dataset_loader = Planetoid(root=root_data_dir, name=dataset_name)
        data_object = dataset_loader[0] # Get the single graph object from the dataset.

        # --- Step 2: Validate the loaded data ---
        if data_object.x is None or data_object.edge_index is None:
            print(f" ERROR: Loaded data for '{dataset_name}' is invalid (missing features or edges).")
            return False

        print(f"  '{dataset_name}' loaded successfully.")
        print(f"    - Nodes: {data_object.num_nodes}")
        print(f"    - Edges: {data_object.num_edges}")
        print(f"    - Node Features: {dataset_loader.num_node_features}")
        print(f"    - Classes: {dataset_loader.num_classes}")

        # --- Step 3: Create a dictionary to be saved ---
        save_obj = {
            'data': data_object,
            'num_node_features': dataset_loader.num_node_features,
            'num_classes': dataset_loader.num_classes,
            'name': dataset_name
        }

        # --- Step 4: Define the save path and save the object ---
        save_filename = f"{dataset_name}_processed.pt"
        save_path = os.path.join(processed_data_output_dir, save_filename)
        
        torch.save(save_obj, save_path)

        duration = time.time() - start_time
        print(f"  Successfully saved '{dataset_name}' to '{save_path}'")
        print(f"  Processing time: {duration:.2f} seconds.\n")
        return True

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n  *** FATAL ERROR while processing '{dataset_name}': {e}")
        print(f"  Processing time before error: {duration:.2f} seconds.\n")
        traceback.print_exc()
        return False

# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Starting Dataset Pre-processing Script ---")

    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    root_data_dir_abs = os.path.abspath(os.path.join(script_dir, DATA_DIR))
    processed_data_output_dir_abs = os.path.abspath(os.path.join(script_dir, PROCESSED_DATA_DIR))

    # Create directories 
    os.makedirs(root_data_dir_abs, exist_ok=True)
    os.makedirs(processed_data_output_dir_abs, exist_ok=True)

    print(f"Root directory for raw datasets: {root_data_dir_abs}")
    print(f"Output directory for processed files: {processed_data_output_dir_abs}")
    print("-" * 50)

    success_count = 0
    fail_count = 0
    for ds_name in DATASETS_TO_PREPROCESS:
        if preprocess_and_save(ds_name, root_data_dir_abs, processed_data_output_dir_abs):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 50)
    print("--- Pre-processing Summary ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process:    {fail_count}")

    if fail_count > 0:
        print("\n WARNING: One or more datasets failed. Please review the errors above.")
    else:
        print("\nAll datasets processed successfully.")
    
 

