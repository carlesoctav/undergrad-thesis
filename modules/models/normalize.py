class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, features: Tensor) -> Tensor:
        features_normalized = F.normalize(features, p=2, dim=1)
        return features_normalized
