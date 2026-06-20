import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class BugScopeGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def get_risk_level(score):
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    return "LOW"


def build_llm_context(metrics_dict):
    context = []

    if metrics_dict.get("cyclomatic_complexity", 0) >= 10:
        context.append("High cyclomatic complexity")

    if metrics_dict.get("avg_function_length", 0) >= 40:
        context.append("Long average function length")

    if metrics_dict.get("comment_density", 0) < 0.05:
        context.append("Low comment density")

    if metrics_dict.get("code_churn", 0) >= 80:
        context.append("High code churn")

    if metrics_dict.get("out_degree", 0) >= 3:
        context.append("Imports many internal modules")

    if metrics_dict.get("in_degree", 0) >= 3:
        context.append("Used by many internal modules")

    if metrics_dict.get("pagerank", 0) >= 0.2:
        context.append("High graph importance")

    return context


def predict_with_gnn(result):
    checkpoint = torch.load(
        "Engine/bugscope_gcn_checkpoint.pt",
        map_location="cpu"
    )

    model = BugScopeGCN(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor(result["metrics"]["X"], dtype=torch.float32)

    edge_index = torch.tensor(
        result["metrics"]["edge_index"],
        dtype=torch.long
    )

    if edge_index.numel() > 0:
        reverse_edge_index = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)

    data = Data(x=x, edge_index=edge_index)

    file_paths = [node["id"] for node in result["nodes"]]

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

    feature_names = result["metrics"]["feature_names"]
    raw_features = result["metrics"]["features"]

    # adjacency list for affected files
    adj = {}
    for src, dst in zip(edge_index[0], edge_index[1]):
        src = int(src)
        dst = int(dst)

        if src not in adj:
            adj[src] = set()

        adj[src].add(dst)

    report = []

    for i, source_score in enumerate(probs):
        source_score = float(source_score)
        file_path = file_paths[i]

        feature_vector = raw_features[file_path]
        feature_map = dict(zip(feature_names, feature_vector))

        important_metrics = {
            "loc": feature_map.get("loc", 0),
            "cyclomatic_complexity": feature_map.get("cyclomatic_complexity", 0),
            "avg_function_length": feature_map.get("avg_function_length", 0),
            "comment_density": feature_map.get("comment_density", 0),
            "code_churn": feature_map.get("code_churn", 0),
            "commit_frequency": feature_map.get("commit_frequency", 0),
            "num_of_developers": feature_map.get("num_of_developers", 0),
            "in_degree": feature_map.get("in_degree", 0),
            "out_degree": feature_map.get("out_degree", 0),
            "total_degree": feature_map.get("total_degree", 0),
            "dependency_ratio": feature_map.get("dependency_ratio", 0),
            "pagerank": feature_map.get("pagerank", 0),
        }

        affected_files = []

        for neighbor in adj.get(i, []):
            neighbor_score = float(probs[neighbor])
            affected_score = source_score * neighbor_score

            affected_files.append({
                "file": file_paths[neighbor],
                "risk_score": round(neighbor_score, 4),
                "affected_score": round(affected_score, 4)
            })

        affected_files = sorted(
            affected_files,
            key=lambda x: x["affected_score"],
            reverse=True
        )

        report.append({
            "file": file_path,
            "risk_score": round(source_score, 4),
            "risk_level": get_risk_level(source_score),
            "prediction": 1 if source_score >= 0.5 else 0,
            "important_metrics": important_metrics,
            "affected_files": affected_files,
            "llm_context": build_llm_context(important_metrics)
        })

    report = sorted(
        report,
        key=lambda x: x["risk_score"],
        reverse=True
    )

    return report