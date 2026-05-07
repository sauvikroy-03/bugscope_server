from sklearn.preprocessing import StandardScaler
import torch
def convert_to_gnn_format(final_features,index_map,edge_index):

    #Step 1: Convert features into X 
    X = [None] * len(index_map)
    for node_id,index in index_map.items():
        X[index]=final_features[node_id]

    #Apply Standardisation

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # return (X_scaled.tolist())

    #Convert to Tensors
    x = torch.tensor(X_scaled, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return{
        "X":x.tolist(),
        "edge_index":edge_index.tolist()
    }
    
