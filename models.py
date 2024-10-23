import torch
from torch import seed_everything
from torch_geometric.nn import SAGEConv

seed_everything(42, workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = SAGEConv((-1, -1), 4 * out_channels)
        self.conv2 = SAGEConv(4 * out_channels, 2 * out_channels)
        # self.conv3 = SAGEConv(4 * out_channels, 2 * out_channels)
        self.conv4 = SAGEConv(2 * out_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.linear(x)
        # x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
        return x


""" Este es el modelo bueno
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = SAGEConv((-1, -1), 4 * out_channels)
        self.conv2 = SAGEConv(4 * out_channels, 2 * out_channels)
        self.conv3 = SAGEConv(2 * out_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


out_channels = 16
"""
