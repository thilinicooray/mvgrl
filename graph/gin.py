import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid,Tanh
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        #self.embedding = Sequential(Linear(num_features, dim), ReLU())


        for i in range(num_gc_layers+4):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

        '''
        # node
        self.node_mu = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_logvar = Linear(in_features=dim, out_features=dim, bias=True)

        # class
        self.class_mu = Linear(in_features=dim, out_features=dim, bias=True)
        self.class_logvar = Linear(in_features=dim, out_features=dim, bias=True)'''


    def forward(self, x, edge_index, batch):

        '''if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)'''

        #print('input x' , x[:5])

        #x = self.embedding(x)
        dropout_val = 0.3

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            #x = F.dropout(x, dropout_val, training=self.training)
            xs.append(x)
            # if i == 2:
                # feature_map = x2
        j = self.num_gc_layers
        #self.bns[j](
        node_latent_space_mu = self.bns[j](torch.tanh(self.convs[j](x, edge_index)))

        #print('node mu bn', self.bns[j].running_mean, self.bns[j].running_var)


        node_latent_space_logvar = self.bns[j+1](torch.tanh(self.convs[j+1](x, edge_index)))

        class_latent_space_mu = self.bns[j+2](torch.tanh(self.convs[j+2](x, edge_index)))
        class_latent_space_logvar = self.bns[j+3](torch.tanh(self.convs[j+3](x, edge_index)))

        '''node_latent_space_mu = F.relu(self.node_mu(x))
        node_latent_space_logvar = F.relu(self.node_logvar(x))

        class_latent_space_mu = F.relu(self.class_mu(x))
        class_latent_space_logvar = F.relu(self.class_logvar(x))'''


        return node_latent_space_mu, node_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar


class Decoder(torch.nn.Module):
    def __init__(self, node_dim, class_dim, feat_size):
        super(Decoder, self).__init__()

        self.linear_model = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(in_features=node_dim + class_dim, out_features=node_dim, bias=True)),
            ('relu_1', ReLU()),

            ('linear_2', torch.nn.Linear(in_features=node_dim, out_features=feat_size, bias=True)),
            ('relu_final', Tanh()),
        ]))

    def forward(self, node_latent_space, class_latent_space, edge_index):
        x = torch.cat((node_latent_space, class_latent_space), dim=1)

        x = torch.softmax(self.linear_model(x), dim=-1)
        #x = self.linear_model(x)

        #value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        #return torch.sigmoid(value)

        return x



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print(data.x.shape)
        # [ num_nodes x num_node_labels ]
        # print(data.edge_index.shape)
        #  [2 x num_edges ]
        # print(data.batch.shape)
        # [ num_nodes ]
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':

    for percentage in [ 1.]:
        for DS in [sys.argv[1]]:
            if 'REDDIT' in DS:
                epochs = 200
            else:
                epochs = 100
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
            accuracies = [[] for i in range(epochs)]
            #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
            dataset = TUDataset(path, name=DS) #.shuffle()
            num_graphs = len(dataset)
            print('Number of graphs', len(dataset))
            dataset = dataset[:int(num_graphs * percentage)]
            dataset = dataset.shuffle()

            kf = KFold(n_splits=10, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(dataset):

                # x_train, x_test = x[train_index], x[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                train_dataset = [dataset[int(i)] for i in list(train_index)]
                test_dataset = [dataset[int(i)] for i in list(test_index)]
                print('len(train_dataset)', len(train_dataset))
                print('len(test_dataset)', len(test_dataset))

                train_loader = DataLoader(train_dataset, batch_size=128)
                test_loader = DataLoader(test_dataset, batch_size=128)
                # print('train', len(train_loader))
                # print('test', len(test_loader))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, epochs+1):
                    train_loss = train(epoch)
                    train_acc = test(train_loader)
                    test_acc = test(test_loader)
                    accuracies[epoch-1].append(test_acc)
                    tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                       train_acc, test_acc))
            tmp = np.mean(accuracies, axis=1)
            print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
            input()
