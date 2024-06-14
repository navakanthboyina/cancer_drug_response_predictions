from torch.utils.data import Dataset,TensorDataset
from src import *
from utils.util import *
from drug_dae import DAE
cell_dataset = CustomDataset("../data/cell2mutation.txt")
from torch import nn
import torch
# DataLoader is used to load the dataset
# for training
cell_loader = torch.utils.data.DataLoader(TensorDataset(torch.Tensor(cell_dataset).to(DEVICE)),
                                     batch_size=32,
                                     shuffle=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get initial iteration of new training dl
new_train_dl = cell_loader

visible_dim = CELL_DIMENSION
hidden_dim = None
models = []  # trained RBM models
for configs in drug_hidden_dimensions:

    # parse configs
    hidden_dim = configs["hidden_dim"]
    num_epochs = configs["num_epochs"]
    lr = configs["learning_rate"]
    use_gaussian = configs["use_gaussian"]

    print(configs)
    # train RBM
    print(f"{visible_dim} to {hidden_dim}")
    model, v, v_pred = train_rbm(new_train_dl, visible_dim, hidden_dim, k=1, num_epochs=num_epochs, lr=lr,
                                 use_gaussian=use_gaussian)
    print("model execution_completed")
    models.append(model)

    # rederive new data loader based on hidden activations of trained model
    new_data = []
    for data_list in new_train_dl:
        p = model.sample_h(data_list[0])[0]
        new_data.append(p.detach().cpu().numpy())
    new_input = np.concatenate(new_data)
    new_train_dl = DataLoader(
        TensorDataset(torch.Tensor(new_input).to(DEVICE)),
        batch_size=32,
        shuffle=False
    )
    print(new_train_dl)

    # update new visible_dim for next RBM
    visible_dim = hidden_dim
    print(visible_dim)


# fine-tune autoencoder
lr = 1e-3
dae = DAE(models).to(DEVICE)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(dae.parameters(), lr)
num_epochs = 50

# train
epoch_loss = []
for epoch in range(num_epochs):
    losses = []
    for i, data_list in enumerate(cell_loader):
        data = data_list[0]
        v_pred = dae(data)
        batch_loss = loss(data, v_pred) # difference between actual and reconstructed
        losses.append(batch_loss.item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    running_loss = np.mean(losses)
    print(f"Epoch {epoch}: {running_loss}")
    epoch_loss.append(running_loss)

torch.save(dae, "../model/rbm_cell_1"+ str(num_epochs))

# plt.style.use('fivethirtyeight')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Error plot for {}'.format("GENOTYPE_RBM_AE"))
plt.legend("Train MSE error")
plt.plot(epoch_loss[-1 * num_epochs:])
plt.savefig("cell_rbm_loss" + ".png")
plt.show()

plt.plot(epoch_loss)
plt.savefig("cell_rbm_loss_1" + ".png")
plt.show()