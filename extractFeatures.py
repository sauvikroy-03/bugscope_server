import gnn
def build_graph_metrics(graph,code_features,gitFeatures):
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Map node id → index
    node_ids = [n["id"] for n in nodes]

    # its goes like {"fileA.py": 0, "fileB.py": 1, ...}
    index_map = {}
    for i, node_id in enumerate(node_ids):
        index_map[node_id] = i

    # Degree
    in_degree = {node_id: 0 for node_id in node_ids}
    out_degree = {node_id: 0 for node_id in node_ids}

    # Edge index (for PyTorch Geometric later)
    edge_index = [[], []]

    # Fill degrees + edge_index
    for e in edges:
        src = e["source"]
        tgt = e["target"]

        if src not in index_map or tgt not in index_map:
            continue

        i = index_map[src]
        j = index_map[tgt]

        edge_index[0].append(i)
        edge_index[1].append(j)

        out_degree[src] += 1
        in_degree[tgt] += 1

    # Depth (folder hierarchy)
    depth = {
    node_id: node_id.replace("\\", "/").count("/")
    for node_id in node_ids
}

    # Total degree
    total_degree = {
        node_id: in_degree[node_id] + out_degree[node_id]
        for node_id in node_ids
    }

    # Dependency ratio (0 → source(all out no in degree), 1 → sink(all in no out degree ,means heavily dependednt module))
    # gives value between 0 and 1 indicating how much a file is a dependency (1 means it's only imported, 0 means it only imports others)
    #0.5 means balanced, more than 0.5 means more of a dependency, less than 0.5 means more of an importer
    ratio = {
        node_id: in_degree[node_id] / (in_degree[node_id] + out_degree[node_id])
        if (in_degree[node_id] + out_degree[node_id]) > 0 else 0.0
        for node_id in node_ids
    }

    # Feature schema (IMPORTANT)
    feature_names = [
    "in_degree",
    "out_degree",
    "total_degree",
    "depth",
    "is_leaf",
    "is_root",
    "dependency_ratio",
    "loc",
    "functions",
    "classes",
    "loops",
    "if_else",
    "cyclomatic_complexity",
    "avg_function_length",
    "comment_density",
    "num_of_developers",
    "code_churn",
    "commit_frequency"

]

    # Build features
    features = {
        node_id: [
            in_degree[node_id],
            out_degree[node_id],
            total_degree[node_id],
            depth[node_id],#how much deep is the file eg- a/b/main.py
            1 if out_degree[node_id] == 0 else 0, #calculating if leaf node
            1 if in_degree[node_id] == 0 else 0, #calculate if root node
            ratio[node_id]
        ]
        for node_id in node_ids
    }

    final_features = {}


    for node_id in node_ids:
        graph_feat = features[node_id]
        node_key = node_id.replace("\\", "/") #to assign correct file path as key earlier it was node_id
        code_feat = code_features.get(node_key,code_features.get(node_id, [0, 0, 0, 0, 0, 0, 0, 0]))
        '''
        Meaning:

            first try "server/server.py"(normalized path with forward slashes[node_key])
            if not found, try "server\\server.py"
           if still not found, use zeros

            Same for git features:
        '''
        git_feat = gitFeatures.get(node_key,gitFeatures.get(node_id, [0, 0, 0]))
                                                                             #we can do this code_feat = code_features[node_id] if we are sure that every node_id will be present in code_features but to be safe we are using get method with default value as list of 0s

        final_features[node_id] = graph_feat + code_feat+git_feat
    
    gnnConverted=gnn.convert_to_gnn_format(final_features,index_map,edge_index)
    return {
        "features": final_features,
        "feature_names": feature_names,
        "edge_index": edge_index,
        "index_map": index_map,
        "X":gnnConverted["X"],
    }