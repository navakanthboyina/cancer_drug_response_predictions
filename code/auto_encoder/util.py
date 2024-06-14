import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from src.rbm import RBM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_mapping(mapping_file):
    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()

    return mapping


def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')

            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([float(tokens[2])])

    return feature, label


def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label), cell2id_mapping, drug2id_mapping)


class CustomDataset(Dataset):
    """drug  and genotype embedding"""

    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file, header=None).to_numpy()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.data_frame[idx]


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def vae_train_model(model,loader,epochs,name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)
    model = model.to(device)
    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,momentum=0.9)
    loss_function = torch.nn.BCELoss()
    outputs = []
    losses = []
    for epoch in tqdm(range(epochs)):
        for data in loader:
            data = data.to(device)
            reconstructed, mu, logvar = model(data)
            bce_loss = loss_function(reconstructed, data)
            loss = final_loss(bce_loss, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            #print(loss)
            #print(mu,logvar)
            outputs.append((epochs, data, reconstructed))
    torch.save(model, "../model/" + name + str(epochs))
    return losses



def train_model(model,loader,epochs,name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used:",device)
    model = model.to(device)
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr=1e-1,
                                 momentum=0.9)

    outputs = []
    epoch_loss = []
    for epoch in tqdm(range(epochs)):
        losses = []
        for data in loader:
            data = data.float()
            data = data.to(device)
            print("-------",data.dtype,"-------")
            print(data.shape)
            encoded, reconstructed = model(data)
            print("encoder", encoded.dtype, "recon ", reconstructed.dtype)
            #print(reconstructed)
            # Calculating the loss function
            loss = loss_function(reconstructed, data)
            print(loss.dtype)
            loss = loss.float()
            print("after: ",loss.dtype)

            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.cpu().detach().numpy()
)
            # print(loss)
            outputs.append((epochs, data, reconstructed))
        running_loss = np.mean(losses)
        epoch_loss.append(np.mean(losses))
        #print("name: ",model.dtype())
    torch.save(model, "/content/drive/MyDrive/FA22 MS DS/Semester_2/SP23 MACHINE LEARNING BIOINFORMATCS 4062/Project/project_code/project_code/output/model/" + name + str(epochs))
    return epoch_loss

def plot(losses,value,name):
    for i in range(len(losses)):
      losses[i] = losses[i]
    #plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses[-1*value:])
    plt.savefig(name+".png")
    plt.show()


def get_encodings_vae(path,x):
    model = torch.load(path)
    with torch.no_grad():
        rec,mu,var = model(x)
    return mu+var

def get_encodings_rbm_vae(path,x):
    model = torch.load(path)
    with torch.no_grad():
        k = model.encode(x)
    return k
def get_encodings_auto_encoder(path,x):
    model  = torch.load(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = model.to(device)
    with torch.no_grad():
        k = model.encoder(x)
    print(k)
    return k


def train_rbm(train_dl, visible_dim, hidden_dim, k, num_epochs, lr, use_gaussian=False):
    """Create and train an RBM

    Uses a custom strategy to have 0.5 momentum before epoch 5 and 0.9 momentum after

    Parameters
    ----------
    train_dl: DataLoader
        training data loader
    visible_dim: int
        number of dimensions in visible (input) layer
    hidden_dim: int
        number of dimensions in hidden layer
    k: int
        number of iterations to run for Gibbs sampling (often 1 is used)
    num_epochs: int
        number of epochs to run for
    lr: float
        learning rate
    use_gaussian:
        whether to use a Gaussian distribution for the hidden state

    Returns
    -------
    RBM, Tensor, Tensor
        a trained RBM model, sample input tensor, reconstructed activation probabilities for sample input tensor
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rbm = RBM(visible_dim=visible_dim, hidden_dim=hidden_dim, gaussian_hidden_distribution=use_gaussian)
    loss = torch.nn.MSELoss()  # we will use MSE loss

    for epoch in range(num_epochs):
        train_loss = 0
        for i, data_list in enumerate(train_dl):
            sample_data = data_list[0].to(DEVICE)

            v0, pvk = sample_data, sample_data
            #print("v0 : ",v0.shape,"pvk: ", pvk.shape)
            # Gibbs sampling
            for i in range(k):
                _, hk = rbm.sample_h(pvk)
                pvk = rbm.sample_v(hk)

            # compute ph0 and phk for updating weights
            ph0, _ = rbm.sample_h(v0)
            phk, _ = rbm.sample_h(pvk)

            # update weights
            rbm.update_weights(v0, pvk, ph0, phk, lr,
                               momentum_coef=0.5 if epoch < 5 else 0.9,
                               weight_decay=2e-4,
                               batch_size=sample_data.shape[0])

            # track loss
            train_loss += loss(v0, pvk)

        # print training loss
        print(f"epoch {epoch}: {train_loss / len(train_dl)}")
    return rbm, v0, pvk