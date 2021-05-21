import torch
from torch import nn

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool

model = Sequential(
    "x, edge_index, batch",
    [
        (Dropout(p=0.5), "x -> x"),
        ReLU(inplace=True),
        (GCNConv(784, 64), "x, edge_index -> x1"),
        (GCNConv(64, 64), "x1, edge_index -> x2"),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], "x1, x2 -> xs"),
        (JumpingKnowledge("cat", 64, num_layers=2), "xs -> x"),
        (global_mean_pool, "x, batch -> x"),
        Linear(2 * 64, 10),
    ],
)


def summary(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.random.seed()
    x = torch.randn((1, 6, 3, 3))
    depthwise = nn.Conv2d(6, 6, 3, 1, 1, groups=6, bias=False)
    pointwise = nn.Conv2d(6, 9, 1, bias=False)
    p1 = summary(depthwise)
    p2 = summary(pointwise)
    print(f"Total parameters: {p1 + p2}")


# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-geometric