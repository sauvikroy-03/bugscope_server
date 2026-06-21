from sklearn.preprocessing import StandardScaler
import torch

def convert_to_gnn_format(final_features, index_map, edge_index):
    # Step 1: Normalize keys in final_features to use forward slashes
    clean_features = {str(k).replace("\\", "/"): v for k, v in final_features.items()}
    
    # Step 2: Convert features into X matching the index_map positions
    X = [None] * len(index_map)
    for node_id, index in index_map.items():
        # Standardize the index_map key to forward slashes for the lookup
        clean_node_id = str(node_id).replace("\\", "/")
        
        # Safe lookup with a fallback to zero-vector if key is completely missing
        if clean_node_id in clean_features:
            X[index] = clean_features[clean_node_id]
        else:
            # Fallback to a zero vector of the same feature length to prevent None errors
            first_val = next(iter(clean_features.values()))
            X[index] = [0.0] * len(first_val)
            print(f"⚠️ Warning: Missing features for standardized path: {clean_node_id}")

    # Apply Standardisation safely now that X has no unaligned positions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to Tensors
    x = torch.tensor(X_scaled, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return {
        "X": x.tolist(),
        "edge_index": edge_index.tolist()
    }