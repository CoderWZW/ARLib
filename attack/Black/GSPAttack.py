import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tool import targetItemSelect
import scipy.sparse as sp
from scipy.sparse import vstack,csr_matrix
from recommender.NGCF import NGCF
import math
import argparse
from conf.recommend_parser import recommend_parse_args

class GSPAttack():
    def __init__(self, arg, data):
        """
        generate fake users by GAN
        :param arg: parameter configuration
        :param data: dataLoader
        """
        self.args=recommend_parse_args()
        self.interact = data.matrix()
        self.userNum = data.user_num
        self.itemNum = data.item_num
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.maliciousUserSize = arg.maliciousUserSize
        self.targetSize = arg.targetSize
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]


        # The probability that non-target items are sampled
        self.itemP = np.array((self.interact.sum(0) / self.interact.sum()))[0]
        self.itemP[self.targetItem] = 0
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)
        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch
        self.alpha = 1
        self.beta = 1
        self.batchSize = 128
        self.ngcf = NGCFProxy(data, 64, 2, self.fakeUserNum, self.maliciousFeedbackNum).cuda()

    def posionDataAttack(self):
        recommender = self.ngcf
        optimizer = torch.optim.Adam(recommender.parameters(), lr=0.01)
        # recommender.train(Epoch=50, optimizer=optimizer_preTrain, evalNum=5)
        bestLoss = 10e10
        for epoch in range(self.Epoch):
            Pu, Pi = recommender()
            scores = torch.zeros((self.userNum + self.fakeUserNum, self.itemNum))
            for batch in range(0,self.userNum + self.fakeUserNum, self.batchSize):
                scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                @ Pi.T)
            L_per = 0
            with torch.no_grad():
                uiAdj = recommender.adj_mat.to_dense()[:self.userNum+self.fakeUserNum,:self.itemNum].cuda()
            k = 0
            for batch in range(0,self.userNum + self.fakeUserNum, self.batchSize):
                k += 1
                L_per += -(uiAdj[batch:batch + self.batchSize, :] * torch.log(F.sigmoid(scores[batch:batch + self.batchSize, :]).cuda() + 10e-8)+
                (1 - uiAdj[batch:batch + self.batchSize, :]) * (torch.log(1 - F.sigmoid(scores[batch:batch + self.batchSize, :].cuda()) + 10e-8))).mean()
            L_per = L_per / k
            L_exPR = 0
            k = 0
            for i in range(self.userNum, self.userNum + self.fakeUserNum):
                for j in self.targetItem:
                    k += 1
                    L_exPR += -torch.log(F.sigmoid(scores[i,j]) + 10e-8)
            L_exPR = L_exPR / k
            Loss = self.alpha * L_per + self.beta * L_exPR
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            print("loss Value:{}".format(Loss))

            if Loss < bestLoss:
                bestAdj = recommender.adj_mat
                bestLoss = Loss
            print("BiLevel epoch {} is over\n".format(epoch + 1))
        coo = bestAdj.coalesce().cpu().indices().numpy()
        bestAdj = csr_matrix((bestAdj.coalesce().values().cpu().numpy(),(coo[0],coo[1])), \
                shape=(self.userNum+self.fakeUserNum, self.itemNum), dtype=np.float32)
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        bestAdj[self.fakeUser, :] = self.project(bestAdj[self.fakeUser, :],self.maliciousFeedbackNum)
        for u in self.fakeUser:
            bestAdj[u,self.targetItem] = 1
        self.interact = bestAdj
        return self.interact
    
    def project(self, mat, n):
        try:
            matrix = torch.tensor(mat[:, :].todense())
            _, indices = torch.topk(matrix, n, dim=1)
            matrix.zero_()
            matrix.scatter_(1, indices, 1)
        except:
            matrix = mat[:,:]
            for i in range(matrix.shape[0]):
                subMatrix = torch.tensor(matrix[i, :].todense())
                topk_values, topk_indices = torch.topk(subMatrix, n)
                subMatrix.zero_()  
                subMatrix[0, topk_indices] = 1
                matrix[i, :] = subMatrix[:, :].flatten()
        return matrix


class MLP(nn.Module):
    def __init__(self, dimIn, dimHid, dimOut=1):
        super(MLP, self).__init__()
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.dimHid = dimHid
        self.net = nn.Sequential(nn.Linear(self.dimIn, self.dimHid),
        nn.ReLU(), nn.Linear(self.dimHid, self.dimOut))
    def forward(self,x):
        out = self.net(x)
        return out

class NGCFProxy(nn.Module):
    def __init__(self, data, emb_size, n_layers, fakeUserNum, maliciousFeedbackNum):
        super(NGCFProxy, self).__init__()
        self.data = data
        self.user_num = self.data.user_num
        self.item_num = self.data.item_num
        self.latent_size = emb_size
        self.layers = n_layers
        self.fakeUserNum = fakeUserNum
        self.mlp = MLP(2 * self.latent_size, int(math.sqrt(self.data.item_num)))
        self.embedding_dict, self.W = self._init_model()
        
        self.adj_mat = self.__create_sparse_torch_adjacency().cuda()
        self.norm_adj = self.__create_sparse_torch_norm_adjacency(self.adj_mat)
        self.Pu = None
        self.maliciousFeedbackNum = maliciousFeedbackNum
    
    def __create_sparse_torch_adjacency(self):
        n_nodes = self.user_num + self.fakeUserNum + self.item_num
        realUserInd = [self.data.user[pair[0]] for pair in self.data.training_data]
        fakeUserInd = [u for u in range(self.user_num, self.user_num + self.fakeUserNum) for _ in range(self.item_num)]
        row_idx = torch.tensor(realUserInd+fakeUserInd)
        col_idx = torch.tensor([self.data.item[pair[1]] for pair in self.data.training_data]+\
                            [i for _ in range(self.user_num, self.user_num + self.fakeUserNum) for i in range(self.item_num)])
        adj_mat = torch.sparse.FloatTensor(indices=torch.stack([row_idx,col_idx]),\
                    values=torch.cat((torch.ones_like(torch.tensor(realUserInd))*1.0,\
                            torch.ones_like(torch.tensor(fakeUserInd))*0.0001),dim=0),size=(n_nodes, n_nodes))
        return adj_mat
    
    def __create_sparse_torch_norm_adjacency(self, adj_mat):
        adj_mat = adj_mat + adj_mat.transpose(0, 1)
        rowsum = torch.tensor(torch.sparse.sum(adj_mat, dim=1))
        d_inv = torch.pow(rowsum.float(), -0.5)
        d_mat_inv = torch.diag(d_inv.to_dense())
        tmp_adj = torch.sparse.mm(adj_mat, d_mat_inv)
        return torch.sparse.mm(d_mat_inv, tmp_adj)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num + self.fakeUserNum, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        w_dict = dict()
        for i in range(self.layers):
            w_dict['w1_' + str(i)] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
            w_dict['w2_' + str(i)] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
        W = nn.ParameterDict(w_dict)
        return embedding_dict, W
    
    def forward(self):
        if self.Pu is None:
            with torch.no_grad():
                ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
                all_embeddings = [ego_embeddings]
                for k in range(self.layers):
                    temp = torch.mm(ego_embeddings, self.W['w1_' + str(k)])
                    ego_embeddings = F.leaky_relu(torch.sparse.mm(self.norm_adj, temp) + \
                                                temp + \
                                                torch.mm(
                                                    torch.sparse.mm(self.norm_adj, ego_embeddings) * ego_embeddings,
                                                    self.W['w2_' + str(k)]))
                    all_embeddings += [ego_embeddings]
                all_embeddings = torch.stack(all_embeddings, dim=1)
                all_embeddings = torch.mean(all_embeddings, dim=1)
                self.Pu = all_embeddings[:self.data.user_num + self.fakeUserNum]
                self.Pi = all_embeddings[self.data.user_num + self.fakeUserNum:]

        for u in range(self.user_num, self.user_num + self.fakeUserNum):
            out = self.mlp(torch.cat((self.Pu[u].repeat(self.item_num, 1),self.Pi), dim=1))
            out2 = self.Gumbel_Softmax(out, self.maliciousFeedbackNum).T
            # self.adj_mat[u,:] = out2
            self.adj_mat._values()[self.adj_mat._indices()[0] == u] = out2
            self.norm_adj = self.__create_sparse_torch_norm_adjacency(self.adj_mat)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            temp = torch.mm(ego_embeddings, self.W['w1_' + str(k)])
            ego_embeddings = F.leaky_relu(torch.sparse.mm(self.norm_adj, temp) + \
                                          temp + \
                                          torch.mm(
                                              torch.sparse.mm(self.norm_adj, ego_embeddings) * ego_embeddings,
                                              self.W['w2_' + str(k)]))
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num + self.fakeUserNum]
        item_all_embeddings = all_embeddings[self.data.user_num + self.fakeUserNum:]
        with torch.no_grad():
            self.Pu = user_all_embeddings
            self.Pi = item_all_embeddings
        return user_all_embeddings, item_all_embeddings
    def Gumbel_Softmax(self, x, k, tau=1):
        mask = torch.ones_like(x)
        out = torch.zeros_like(x)
        for i in range(k):
            out_tmp = F.gumbel_softmax(x * mask,tau=tau)
            mask[range(mask.shape[0]),torch.argmax(out_tmp,dim=1)] = -10e9
            out += out_tmp
        return out
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)