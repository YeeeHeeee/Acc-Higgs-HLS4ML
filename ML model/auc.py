import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score
class SimpleFeedforward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SimpleFeedforward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)

class CouplingLayer(nn.Module):
    def __init__(self, input_dim=140, hidden_layer=2):
        super(CouplingLayer, self).__init__()

        # using as the 'self.coupling == "additive":' 
        self.s_net = SimpleFeedforward(input_dim // 2, input_dim // 2, hidden_layer)
        self.t_net = SimpleFeedforward(input_dim // 2, input_dim // 2, hidden_layer)

    def forward(self, x, reverse=False):
        xa, xb = x.chunk(2, dim=1)

        if not reverse:
            s = torch.sigmoid(self.s_net(xb) + 2)
            t = self.t_net(xb)
            ya = s * xa + t
            y = torch.cat([ya, xb], dim=1)
        else:
            s = torch.sigmoid(self.s_net(xb) + 2)
            t = self.t_net(xb)
            ya = (xa - t) / s
            y = torch.cat([ya, xb], dim=1)

        return y

class INN(nn.Module):
    def __init__(self, input_dim=140, num_coupling_layers=10, hidden_layer=3):
        super(INN, self).__init__()

        self.coupling_layers = nn.ModuleList([
            CouplingLayer(input_dim=input_dim, hidden_layer=hidden_layer)
            for _ in range(num_coupling_layers)
        ])

    def forward(self, x):
        for coupling_layer in self.coupling_layers:
            x = coupling_layer(x)
        return x

    def inverse(self, y):
        for coupling_layer in reversed(self.coupling_layers):
            y = coupling_layer(y, reverse=True)
        return y
model = INN()
model.load_state_dict(torch.load(f'/vols/cms/hw423/Acc/model.pth'))
model.eval()
test_loss = 0.0
y_true = pd.read_parquet('/Acc/df.parquet').reset_index(drop=True)['proc']
x = pd.read_parquet('/Acc/x.parquet')

all_preds = []
all_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        output = model(data)
        # Convert model output and targets to proper format
        preds = output.squeeze().cpu().numpy()  # Adjust for model output specifics
        targets = targets.cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(targets)



auc = roc_auc_score(y_true, all_preds)
print(f'AUC = {auc}')