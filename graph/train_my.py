import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dataset import load


from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder, Decoder
from evaluate_embedding import evaluate_embedding
from model import *
from utils_mine import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence



class GcnInfomax(nn.Module):
    def __init__(self, ft_size, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        #self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(ft_size, hidden_dim, num_gc_layers)
        self.decoder = Decoder(hidden_dim, hidden_dim, ft_size)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)


        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs

        node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)


        '''n_digits = 4
        #print('before ', node_mu)
        node_mu = (node_mu * 10**n_digits).round() / (10**n_digits)
        node_logvar = (node_logvar * 10**n_digits).round() / (10**n_digits)
        class_mu = (class_mu * 10**n_digits).round() / (10**n_digits)
        class_logvar = (class_logvar * 10**n_digits).round() / (10**n_digits)'''


        #print('direct out ', node_mu[0,:5], node_logvar[0,:5], class_mu[0,:5], class_logvar[0,:5], batch)


        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, batch, True
        )

        #print('grouped ', grouped_mu[0,:5], grouped_logvar[0,:5])


        #print('after ', node_mu)


        # kl-divergence error for style latent space
        node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )
        #print('node kl unwei ', node_kl_divergence_loss, node_logvar, node_mu)


        node_kl_divergence_loss = 0.0000001*node_kl_divergence_loss *num_graphs
        #print('node kl wei ', node_kl_divergence_loss)


        # kl-divergence error for class latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        #print('class kl unwei ', class_kl_divergence_loss)
        class_kl_divergence_loss = 0.0000001* class_kl_divergence_loss * num_graphs
        #print('class kl wei ', class_kl_divergence_loss)


        # reconstruct samples
        """
        sampling from group mu and logvar for each graph in mini-batch differently makes
        the decoder consider class latent embeddings as random noise and ignore them 
        """
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        class_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
        )


        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings, edge_index)

        #reconstruction_error =  mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error = 0.01* self.recon_loss(reconstructed_node, edge_index, batch) * num_graphs


        #print(reconstruction_error.item(), class_kl_divergence_loss.item(), node_kl_divergence_loss.item())

        '''n_digits = 9
        #print('before ', node_mu)
        reconstruction_error = (reconstruction_error * 10**n_digits).round() / (10**n_digits)
        class_kl_divergence_loss = (class_kl_divergence_loss * 10**n_digits).round() / (10**n_digits)
        node_kl_divergence_loss = (node_kl_divergence_loss * 10**n_digits).round() / (10**n_digits)'''

        class_kl_divergence_loss.backward(retain_graph=True)
        node_kl_divergence_loss.backward(retain_graph=True)
        reconstruction_error.backward()


        return  reconstruction_error.item(), class_kl_divergence_loss.item() , node_kl_divergence_loss.item()


    def edge_recon(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def recon_loss(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #reco = self.edge_recon(z, edge_index)

        #print('edge recon try ', reco.size(), edge_index.size(), reco [:10], edge_index[0][:10], edge_index[1][:10])


        #recon_adj = self.edge_recon(z, edge_index)

        a, idx_tensor = to_dense_batch(z, batch)
        a_t = a.permute(0, 2, 1)

        #print('batch size', a.size(), a_t.size())

        rec = torch.bmm(a, a_t)

        #print('inner pro', rec.size())

        org_adj = to_dense_adj(edge_index, batch)


        #print('new' ,rec, 'org', org_adj)


        '''#adj = torch.matmul(z, z.t())
  
  
        #r_adj = to_dense_adj(recon_adj, batch)
        #org_adj = to_dense_adj(edge_index, batch)
  
        #print(r_adj.size(), org_adj.size())
  
        recon_adj = self.edge_recon(z, edge_index)
  
  
        pos_loss = -torch.log(
            recon_adj + EPS).mean()
  
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
  
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) #random thingggg
        neg_loss = -torch.log(1 -
                              self.edge_recon(z, neg_edge_index) +
                              EPS).mean()
  
        return pos_loss + neg_loss'''

        loss = F.binary_cross_entropy_with_logits(rec, org_adj)

        return loss


    def recon_loss1(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """



        recon_adj = self.edge_recon(z, edge_index)


        pos_loss = -torch.log(
            recon_adj + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) #random thingggg
        neg_loss = -torch.log(1 -
                              self.edge_recon(z, neg_edge_index) +
                              EPS).mean()

        return pos_loss + neg_loss

        #loss = F.binary_cross_entropy_with_logits(rec, org_adj)

        #return loss

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                #print(x, edge_index, data.x)
                #x = torch.rand(data.batch.shape[0], 5).to(device)
                x = torch.ones((batch.shape[0],5)).double().to(device)
                #print('eval train', x.type())
                __, _, class_mu, class_logvar = self.encoder(x, edge_index, batch)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)

                #print('clz emb ', class_emb[:5,:3])


                ret.append(class_emb.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft, n_h).cuda())
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h).cuda())

    def forward(self, feat, adj, mask):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class Model(nn.Module):
    def __init__(self, n_in, n_h, num_layers):
        super(Model, self).__init__()
        self.mlp1 = MLP(1 * n_h, n_h)
        self.mlp2 = MLP(num_layers * n_h, n_h)
        self.gnn1 = GCN(n_in, n_h, num_layers)
        self.gnn2 = GCN(n_in, n_h, num_layers)

    def forward(self, adj, diff, feat, mask):
        lv1, gv1 = self.gnn1(feat, adj, mask)
        lv2, gv2 = self.gnn2(feat, diff, mask)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2

    def embed(self, feat, adj, diff, mask):
        __, gv1, __, gv2 = self.forward(adj, diff, feat, mask)
        return (gv1 + gv2).detach()


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure, mask):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    for idx, m in enumerate(mask):
        msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t()) * msk

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def train(dataset, gpu, num_layer=4, epoch=40, batch=64):
    nb_epochs = epoch
    batch_size = batch
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj, diff, feat, labels, num_nodes = load(dataset)

    feat = torch.FloatTensor(feat).cuda()
    diff = torch.FloatTensor(diff).cuda()
    adj = torch.FloatTensor(adj).cuda()
    labels = torch.LongTensor(labels).cuda()

    ft_size = feat[0].shape[1]
    max_nodes = feat[0].shape[0]


    model = GcnInfomax(ft_size, hid_units, 2).to(device)








    #model = Model(ft_size, hid_units, num_layer)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    #model.cuda()

    cnt_wait = 0
    best = 1e9

    itr = (adj.shape[0] // batch_size) + 1
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        train_idx = np.arange(adj.shape[0])
        np.random.shuffle(train_idx)

        recon_loss_all = 0
        kl_class_loss_all = 0
        kl_node_loss_all = 0
        mi_loss_all = 0

        for idx in range(0, len(train_idx), batch_size):
            model.train()
            optimiser.zero_grad()

            batch = train_idx[idx: idx + batch_size]
            mask = num_nodes[idx: idx + batch_size]

            recon_loss, kl_class, kl_node = model(feat[batch], adj[batch], batch, )
            recon_loss_all += recon_loss
            kl_class_loss_all += kl_class
            kl_node_loss_all += kl_node





            lv1, gv1, lv2, gv2 = model(adj[batch], diff[batch], feat[batch], mask)

            lv1 = lv1.view(batch.shape[0] * max_nodes, -1)
            lv2 = lv2.view(batch.shape[0] * max_nodes, -1)

            batch = torch.LongTensor(np.repeat(np.arange(batch.shape[0]), max_nodes)).cuda()

            loss1 = local_global_loss_(lv1, gv2, batch, 'JSD', mask)
            loss2 = local_global_loss_(lv2, gv1, batch, 'JSD', mask)
            # loss3 = global_global_loss_(gv1, gv2, 'JSD')
            loss = loss1 + loss2 #+ loss3
            epoch_loss += loss
            loss.backward()
            optimiser.step()








        '''epoch_loss /= itr

        # print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, epoch_loss))

        if epoch_loss < best:
            best = epoch_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), f'{dataset}-{gpu}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

    model.load_state_dict(torch.load(f'{dataset}-{gpu}.pkl'))

    features = feat.cuda()
    adj = adj.cuda()
    diff = diff.cuda()
    labels = labels.cuda()

    embeds = model.embed(features, adj, diff, num_nodes)

    x = embeds.cpu().numpy()
    y = labels.cpu().numpy()

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print(np.mean(accuracies), np.std(accuracies))'''


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    gpu = 1
    #torch.cuda.set_device(gpu)
    layers = [2, 8, 12]
    batch = [32, 64, 128, 256]
    epoch = [20, 40, 100]
    ds = ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    seeds = [123, 132, 321, 312, 231]
    for d in ds:
        print(f'####################{d}####################')
        for l in layers:
            for b in batch:
                for e in epoch:
                    for i in range(5):
                        seed = seeds[i]
                        torch.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        np.random.seed(seed)
                        print(f'Dataset: {d}, Layer:{l}, Batch: {b}, Epoch: {e}, Seed: {seed}')
                        train(d, gpu, l, e, b)
        print('################################################')