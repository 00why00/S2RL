import os

import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

base_dir = os.environ.get("S2RL_FEATS_ROOT", "feats")
for i in tqdm(range(150)):
    mean_feats = torch.load(os.path.join(base_dir, "dinov3_ms", f"mean_feats_{i}.pt"))

    if mean_feats.shape[0] <= 50:
        torch.save(mean_feats, os.path.join(base_dir, "dinov3_kmeans", f"feats_{i}.pt"))
    else:
        kmeans = KMeans(n_clusters=50, random_state=0).fit(mean_feats.cpu().numpy())
        centroids = torch.tensor(kmeans.cluster_centers_).to(mean_feats.device)
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
        sim = torch.mm(centroids, mean_feats.t())
        indices = torch.argmax(sim, dim=1)
        selected_mean = mean_feats[indices]
        torch.save(selected_mean, os.path.join(base_dir, "dinov3_kmeans", f"feats_{i}.pt"))
