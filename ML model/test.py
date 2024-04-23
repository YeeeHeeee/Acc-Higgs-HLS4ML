
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplhep as hep
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', 150)

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
hep.style.use("CMS")
filecode = '128x6'

class Net(nn.Module):
    def __init__(self, n_features=50, nodes=[100,100], output_nodes=6):
        super(Net, self).__init__()
        # Build network
        n_nodes = [n_features] + nodes + [output_nodes]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes)-1):
            self.layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        # Apply softmax
        return torch.softmax(out, dim=1)

train_hp = {
    "lr":0.0001,
    "batch_size":10000,
    "N_epochs":300,
    "seed":0
}
MInum = 140
model_bce = Net(n_features=MInum, nodes=[128,128,128,128,128,128], output_nodes=7)

# # df = pd.read_parquet(f'/vols/cms/hw423/Data/Week6/df_balanced.parquet')
# # df = pd.read_parquet('/vols/cms/hw423/Data/DF.parquet')
# df = df.dropna().reset_index(drop=True)
# y = df['proc']
# dfx = df.iloc[:,:-1]
# dfw = df['weight']*138000
# print('Data input finished')

# dfx.dropna()

# dfx_new = pd.DataFrame(columns = dfx.columns)
# for col in dfx.columns:
#     x = dfx[col]
#     x9 = x[x!=-999]
#     mean = x9.mean()
#     sigma = x9.std()
#     if sigma !=0:
#         x9z = (x9-mean)/sigma
#     else:
#         x9z = x9-mean
#     x_new = x
#     x_new[x!=-999] = x9z.values.astype('float32')
#     x_new[x==-999] = -5
#     dfx_new[col]=x_new
# dfx_new = dfx_new.dropna()
# print('Data normalisation finished')

# del dfx

# X_train = pd.read_parquet('Week3/datas/X_train.parquet')
mi_series = pd.read_csv(f'/vols/cms/hw423/Week6/MI_balanced.csv')
MIcol = mi_series.head(MInum)['Features']
# dfx_new=dfx_new[MIcol]
model_bce.load_state_dict(torch.load(f'/vols/cms/hw423/Data/Week6/model_bce_unblcd_{filecode}.pth'))
train_loss_bce = np.load(f'/vols/cms/hw423/Data/Week6/train_loss_bce_{filecode}.npy')
test_loss_bce = np.load(f'/vols/cms/hw423/Data/Week6/test_loss_bce_{filecode}.npy')


# Plot loss function curves
fig, ax = plt.subplots()
print(train_loss_bce.shape)
ax.plot(np.linspace(0,train_loss_bce.shape[0]-1,train_loss_bce.shape[0]), train_loss_bce, label="Train")
ax.plot(np.linspace(0,train_loss_bce.shape[0]-1,train_loss_bce.shape[0]), test_loss_bce, label="Test")
ax.legend(loc='best')
ax.set_xlabel("Number of epochs")
ax.set_ylabel("BCE loss")
fig.savefig(f'/vols/cms/hw423/Week6/plots/loss_test_wtd_{filecode}.pdf')


col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
oc= model_bce(torch.tensor(dfx_new.to_numpy(),dtype=torch.float32))


bin_num = 40
fig, axs = plt.subplots(2,4,figsize=(25,15))
n = 0
octest =pd.DataFrame(oc.detach(), columns = col, index = df.index)
print(octest)
print(octest.idxmax(axis=1))

for cl in col:
    i = n // 4
    j = n % 4
    otcs = octest[octest.idxmax(axis=1)==cl]
    
    axs[i, j].hist(otcs[df['proc'] == 0][col[0]], weights=dfw[otcs[df['proc'] == 0].index], bins=bin_num, range=(0, 1), color='black', label="$\gamma\gamma$", histtype='step', density=True)
    axs[i, j].hist(otcs[df['proc'] == 1][col[1]], weights=dfw[otcs[df['proc'] == 1].index], bins=bin_num, range=(0, 1), color='blue', label="ggH", histtype='step', density=True)
    axs[i, j].hist(otcs[df['proc'] == 2][col[2]], weights=dfw[otcs[df['proc'] == 2].index], bins=bin_num, range=(0, 1), color='r', label="qqH", histtype='step', density=True)
    axs[i, j].hist(otcs[df['proc'] == 3][col[3]], weights=dfw[otcs[df['proc'] == 3].index], bins=bin_num, range=(0, 1), color='lightseagreen', label="WH", histtype='step', density=True)
    axs[i, j].hist(otcs[df['proc'] == 4][col[4]], weights=dfw[otcs[df['proc'] == 4].index], bins=bin_num, range=(0, 1), color='darkviolet', label="ZH", histtype='step', density=True)
    axs[i, j].hist(otcs[df['proc'] == 5][col[5]], weights=dfw[otcs[df['proc'] == 5].index], bins=bin_num, range=(0, 1), color='gold', label="ttH", histtype='step', density=True)
    axs[i, j].hist(otcs[df['proc'] == 6][col[6]], weights=dfw[otcs[df['proc'] == 6].index], bins=bin_num, range=(0, 1), color='green', label="tH", histtype='step', density=True)
    axs[i, j].legend()
    axs[i, j].set_title(f'Pred = {col[n]}')
    n += 1
print(octest[octest.idxmax(axis=1)==col[0]])
axs[1, 3].axis('off')
plt.tight_layout()
plt.savefig(f'/vols/cms/hw423/Week6/plots/Hist_wtd_{filecode}.pdf')

_, y_pred = torch.max(oc, 1)
print(y_pred)

cm1 = confusion_matrix(y,y_pred,sample_weight=dfw)
cm2 = confusion_matrix(y,y_pred)

# Normalize by row: divide each row by its sum
cm_row_norm = cm1.astype('float') / cm1.sum(axis=1)[:,np.newaxis]
# Normalize by column: divide each column by its sum
cm_col_norm = cm2.astype('float') / cm2.sum(axis=0)[np.newaxis,:]

# Adjusted plotting function that includes saving the figures
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, filename='confusion_matrix_narc.pdf'):
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap)  # Using '.2f' for floating point format
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticklabels(col)
    ax.set_yticklabels(col)
    fig.savefig(filename, bbox_inches='tight')  
    # ax.show()  

# The unnormalized confusion matrix
#plot_confusion_matrix(cm, title='Confusion Matrix (Unnormalized)', filename='Project/New/Week5/plots/Confusion_matrix_unnorm.pdf')

# The row-normalized confusion matrix
plot_confusion_matrix(cm_row_norm, title='Confusion Matrix (Row-normalized)', cmap=plt.cm.Greens, filename=f'/vols/cms/hw423/Week6/plots/Confusion_matrix_row_{filecode}.pdf')

# The column-normalized confusion matrix
plot_confusion_matrix(cm_col_norm, title='Confusion Matrix (Column-normalized, Weighted)', cmap=plt.cm.Oranges, filename=f'/vols/cms/hw423/Week6/plots/Confusion_matrix_col_{filecode}.pdf')
