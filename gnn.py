import torch
import joblib
import os

def convert_to_gnn_format(final_features, index_map, edge_index):
    # Step 1: Normalize keys in final_features to use forward slashes
    clean_features = {str(k).replace("\\", "/"): v for k, v in final_features.items()}
    
    # Step 2: Convert features into X matching the index_map positions
    X = [None] * len(index_map)
    for node_id, index in index_map.items():
        clean_node_id = str(node_id).replace("\\", "/")
        
        if clean_node_id in clean_features:
            X[index] = clean_features[clean_node_id]
        else:
            first_val = next(iter(clean_features.values()))
            X[index] = [0.0] * len(first_val)
            print(f"Warning: Missing features for standardized path: {clean_node_id}")

    # =====================================================================
    # 🌟 FIXED PIPELINE
    # Slices array to isolate the 22 continuous metrics for the global scaler
    # while leaving indices 4 and 5 (is_leaf/is_root flags) raw.
    # =====================================================================
    scaler_path = os.path.join("Engine", "bugscope_scaler.pkl")
    
    try:
        scaler = joblib.load(scaler_path)
        
        # 1. Extract only the 22 continuous columns
        X_continuous = [row[0:4] + row[6:] for row in X]
        
        # 2. Scale them globally
        X_scaled_continuous = scaler.transform(X_continuous)
        
        # 3. Re-stitch the complete 24-feature matrix for GraphSAGE
        X_final = []
        for i, row in enumerate(X):
            scaled_row = list(X_scaled_continuous[i])
            reconstructed = scaled_row[0:4] + [row[4], row[5]] + scaled_row[4:]
            X_final.append(reconstructed)
            
        X_scaled = X_final

    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        X_scaled = X

    # Convert to Tensors
    x = torch.tensor(X_scaled, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return {
        "X": x.tolist(),
        "edge_index": edge_index.tolist()
    }