import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch_geometric.transforms as T
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import HeteroData
from torch_geometric.nn import GAE
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.convert import from_networkx
from torchsummary import summary
from tqdm import tqdm

from .models import GCNEncoder, device

nodes = pd.read_csv("data/nodes.csv")
nodes.drop_duplicates(subset=["protein1", "protein2"], keep="first", inplace=True)
nodes = nodes[nodes.protein2 != 0]
nodes = nodes[nodes.protein1 != 0]
nodes["related"] = nodes["related"].fillna(0).apply(lambda x: 1.0 if x else 0.0)

not_0_nodes = nodes[(nodes.avg_y != 0) & (nodes.avg_o != 0)]
mu = (not_0_nodes.std_y * not_0_nodes.std_o) / (not_0_nodes.std_y + not_0_nodes.std_o)
nodes.loc[not_0_nodes.index, "mu"] = mu
delta_expression = nodes["avg_y"] - nodes["avg_o"]
nodes.loc[:, "delta_expression"] = delta_expression
nodes = nodes.fillna(0)

G = nx.from_pandas_edgelist(nodes, "protein1", "protein2")
G.remove_node(0)
isolated = list(nx.isolates(G))
G.remove_nodes_from(isolated)
pyg_data = from_networkx(G)

protein_to_delta = nodes.set_index("protein1")["delta_expression"].to_dict()
protein_to_mu = nodes.set_index("protein1")["mu"].to_dict()

x = [
    [protein_to_delta.get(node) for node in nx.nodes(G)],
    [protein_to_mu.get(node) for node in nx.nodes(G)],
]

num_features = len(x[0])

protein_data = HeteroData()
protein_data["delta_expression"].x = torch.tensor(x).T
protein_data["delta_expression", "to", "delta_expression"].edge_index = (
    pyg_data.edge_index
)
protein_data["delta_expression"].num_nodes = G.number_of_nodes()
train_mask = torch.tensor(
    [np.random.sample() < 0.7 for _ in range(protein_data.num_nodes)]
)
test_mask = ~train_mask
protein_data["delta_expression"].train_mask = train_mask
protein_data["delta_expression"].test_mask = test_mask
protein_data = T.AddSelfLoops()(protein_data)

pos_edges = protein_data.edge_items()[0][1]["edge_index"].to(device)
neg_edges = negative_sampling(pos_edges, num_features, force_undirected=True).to(device)

protein_data.to(device)
out_channels = 16
epochs = 1000
lr = 1e-2

model = GCNEncoder(num_features, out_channels)
model = to_hetero(model, protein_data.metadata(), aggr="sum")

gae = GAE(encoder=model).to(device)
optimizer = torch.optim.Adam(gae.parameters(), lr=lr)

with torch.no_grad():  # Initialize lazy modules.
    out = gae(protein_data.x_dict, protein_data.edge_index_dict)

loss_history = []
for i, epoch in tqdm(enumerate(range(epochs)), total=epochs):
    gae.train()
    optimizer.zero_grad()
    z = gae(protein_data.x_dict, protein_data.edge_index_dict)
    z = z["delta_expression"]
    loss = gae.recon_loss(z, pos_edges, neg_edges)
    if i % 50 == 0:
        auc, ap = gae.test(z, pos_edges, neg_edges)
        loss_history.append({"loss": float(loss), "auc": auc, "ap": ap})
        print()
        print(loss_history[-1], optimizer.defaults["lr"])
    loss.backward()
    optimizer.step()

df = pd.DataFrame(loss_history)
df.plot()
plt.show()

with torch.no_grad():
    out = gae(protein_data.x_dict, protein_data.edge_index_dict)

print(summary(gae))

u = umap.UMAP(verbose=1, min_dist=0.25, metric="cosine", n_jobs=10, n_neighbors=50)
u.fit(out["delta_expression"].detach().cpu())
vec2 = u.transform(out["delta_expression"].detach().cpu())
df = pd.DataFrame(
    {
        "x": vec2[:, 0],
        "y": vec2[:, 1],
        "delta_expression": protein_data["delta_expression"].x[:, 0].detach().cpu(),
    }
)
df["node"] = list(nx.nodes(G))
df.set_index("node", inplace=True)
nodes_to_join = nodes.drop_duplicates(subset=["protein1", "delta_expression"]).query(
    "protein2 != 0"
)
nodes_to_join.set_index("protein1", inplace=True)
df_joined = df.join(nodes_to_join, how="inner", rsuffix="_join")

plt.figure(figsize=(20, 20))
plt.title(
    """UMAP: verbose=1, min_dist=0.25,
    metric='cosine',
    n_jobs=10,
    n_neighbors=50"""
)
sns.scatterplot(df_joined, x="x", y="y", hue="related", palette="coolwarm")
# plt.savefig('data/vec2.png', dpi=1000)
plt.show()

tsne = TSNE(
    n_components=2,
    learning_rate="auto",
    init="pca",
    metric="cosine",
    n_jobs=10,
    verbose=1,
    perplexity=50,
)

vec2t = tsne.fit_transform(out["delta_expression"].detach().cpu())

df = pd.DataFrame(
    {
        "x": vec2t[:, 0],
        "y": vec2t[:, 1],
        "delta_expression": protein_data["delta_expression"].x[:, 0].detach().cpu(),
    }
)
df["node"] = list(nx.nodes(G))
df.set_index("node", inplace=True)
nodes_to_join = nodes.drop_duplicates(subset=["protein1", "delta_expression"]).query(
    "protein2 != 0"
)
nodes_to_join.set_index("protein1", inplace=True)
df_joined = df.join(nodes_to_join, how="inner", rsuffix="_join")

plt.figure(figsize=(20, 20))
plt.title(
    """TSNE: learning_rate='auto',
    init='pca',
    metric='cosine'
    perplexity=50"""
)
sns.scatterplot(df_joined, x="x", y="y", hue="related", palette="coolwarm")
# plt.savefig('data/tsne_cosine_600_delta.pdf')
plt.show()
