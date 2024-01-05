import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect
import scipy.sparse as sp
from scipy.sparse import vstack,csr_matrix
from recommender.LightGCN import LightGCN
import math
import argparse
from conf.recommend_parser import recommend_parse_args

class LegUP():
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
        self.G = None
        self.D = None
        self.lightgcn = LightGCN(self.args, data)
        self.lightgcn.model = self.lightgcn.model.cuda()

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
        self.BiLevelOptimizationEpoch = 50
        self.Tepoch = 10


    def posionDataAttack(self, epoch1=25, epoch2=25):
        """
        posion Data Generate
        :param epoch: Total epoch
        :param epoch1: discriminator update epoch num
        :param epoch2: generator update epoch num
        """
        if self.G is None:
            self.selectItem = random.sample(set(list(range(self.itemNum)))-set(self.targetItem), self.itemNum//5)  + self.targetItem

            G = Generator(len(self.selectItem)).cpu()
            D = Discriminator(len(self.selectItem)).cpu()
            optimize_G = torch.optim.Adam(G.parameters(), lr=0.005)
            optimize_D = torch.optim.Adam(D.parameters(), lr=0.005)
            for i in range(self.BiLevelOptimizationEpoch):
                G.eval()
                D.train()
                for k1 in range(epoch1):
                    userSet = random.sample(set(list(range(self.userNum))),
                                            self.fakeUserNum)
                    # #sample item rating
                    tempInteract = np.array(
                        [np.random.binomial(1, self.itemP)[self.selectItem] for i in range(self.fakeUserNum)])
                    ind = self.interact[userSet,:].nonzero()
                    row, col, entries = [], [], []
                    for r,c in zip(ind[0].tolist(),ind[1].tolist()):
                        if c not in self.selectItem:continue
                        c = self.selectItem.index(c)
                        row += [r]
                        col += [c]
                        entries += [self.interact[r,c] * tempInteract[r,c]]
                    tempInteract = sp.csr_matrix((entries, (row, col)),
                                                            shape=(len(userSet), len(self.selectItem)),dtype=np.float32)
                    coo = tempInteract.tocoo()
                    inds = torch.LongTensor([coo.row, coo.col])
                    values = torch.from_numpy(coo.data).float()
                    tempInteract = torch.sparse.FloatTensor(inds, values, coo.shape).cpu()
                    fakeInteract = G(tempInteract)
                    loss1 = -(torch.log(D(tempInteract)).mean() + torch.log(1 - D(fakeInteract))).mean()
                    optimize_D.zero_grad()
                    loss1.backward()
                    optimize_D.step()
                    # print("epoch{} miniepoch{} D:{}".format(i, k1, loss1))
                D.eval()
                G.train()
                for k2 in range(epoch2):
                    def fakeUserInject(recommender, user):
                        Pu, Pi = recommender.model()
                        recommender.data.user_num += 1
                        recommender.data.user["fakeuser{}".format(recommender.data.user_num)] = len(
                            recommender.data.user)
                        recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(
                            recommender.data.user_num)

                        row, col, entries = [], [], []
                        for i in self.targetItem:
                            recommender.data.training_data.append(
                                (recommender.data.id2user[user], recommender.data.id2item[i]))
                        for pair in recommender.data.training_data:
                            row += [recommender.data.user[pair[0]]]
                            col += [recommender.data.item[pair[1]]]
                            entries += [1.0]

                        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                                         shape=(recommender.data.user_num,
                                                                                recommender.data.item_num),
                                                                         dtype=np.float32)

                        recommender.__init__(recommender.args, recommender.data)
                        with torch.no_grad():
                            try:
                                recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
                                recommender.model.embedding_dict['item_emb'][:] = Pi
                            except:
                                recommender.model.embedding_dict['user_mf_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0],
                                                                                                :Pu.shape[1] // 2]
                                recommender.model.embedding_dict['user_mlp_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0],
                                                                                                 Pu.shape[1] // 2:]
                                recommender.model.embedding_dict['item_mf_emb'][:] = Pi[:, :Pi.shape[1] // 2]
                                recommender.model.embedding_dict['item_mlp_emb'][:] = Pi[:, Pi.shape[1] // 2:]
                        recommender.model = recommender.model.cuda()

                    for T in range(self.Tepoch):
                        fakeUserInject(self.lightgcn, self.userNum)

                        uiAdj = self.lightgcn.data.matrix()
                        uiAdj2 = uiAdj[:, :]
                        ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                            self.lightgcn.data.user_num + self.itemNum, self.lightgcn.data.user_num + self.itemNum),
                                               dtype=np.float32)
                        ui_adj[:self.lightgcn.data.user_num, self.lightgcn.data.user_num:] = uiAdj2
                        row_indices, col_indices = ui_adj.nonzero()
                        num_samples = np.random.randint(int(self.userNum*0.1),int(self.itemNum*0.1))
                        selected_indices = np.random.choice(len(row_indices), num_samples, replace=False)
                        selected_row_indices = row_indices[selected_indices]
                        selected_col_indices = col_indices[selected_indices]
                        selected_ui_adj = sp.csr_matrix(
                            ([1] * len(selected_row_indices), (selected_row_indices, selected_col_indices)),
                            shape=ui_adj.shape,
                            dtype=np.float32
                        )

                        self.lightgcn.model._init_uiAdj(selected_ui_adj + selected_ui_adj.T)
                        self.lightgcn.train(requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10,
                                            Epoch=0, optimizer=None, evalNum=5)
                        predicted_scores = self.lightgcn.predict(self.lightgcn.data.user_num)

                        target_predicted_scores = predicted_scores[:,
                                                  [self.selectItem.index(item) for item in self.targetItem]]
                        predicted_scores_exp = np.exp(predicted_scores)
                        target_predicted_scores_exp = np.exp(target_predicted_scores)
                        target_predicted_scores_exp = np.expand_dims(target_predicted_scores_exp, axis=2)
                        L_RS = np.sum(target_predicted_scores_exp / predicted_scores_exp)

                    userSet = random.sample(set(list(range(self.userNum))),
                                            self.fakeUserNum)
                    # sample item rating
                    tempInteract = np.array(
                        [np.random.binomial(1, self.itemP)[self.selectItem] for i in range(self.fakeUserNum)])
                    ind = self.interact[userSet,:].nonzero()
                    row, col, entries = [], [], []
                    for r,c in zip(ind[0].tolist(),ind[1].tolist()):
                        if c not in self.selectItem:continue
                        c = self.selectItem.index(c)
                        row += [r]
                        col += [c]
                        entries += [self.interact[r,c] * tempInteract[r,c]]
                    tempInteract = sp.csr_matrix((entries, (row, col)),
                                                            shape=(len(userSet), len(self.selectItem)),dtype=np.float32)
                    coo = tempInteract.tocoo()
                    inds = torch.LongTensor([coo.row, coo.col])
                    values = torch.from_numpy(coo.data).float()
                    tempInteract = torch.sparse.FloatTensor(inds, values, coo.shape).cpu()
                    fakeInteract = G(tempInteract)
                    maskTarget = torch.zeros((len(self.selectItem), 1)).cpu()
                    maskTarget[[self.selectItem.index(i) for i in self.targetItem]] = 1
                    Q = torch.ones_like(fakeInteract)
                    L_GD = torch.log(D(tempInteract)).mean() + torch.log(1 - D(fakeInteract)).mean()
                    loss2 = L_GD +L_RS
                    optimize_G.zero_grad()
                    loss2.backward()
                    optimize_G.step()
                    # print("epoch{} miniepoch{} G:{}".format(i, k2, loss2))
            self.G = G
            self.D = D
        self.G.eval()
        userSet = random.sample(set(list(range(self.userNum))),
                                self.fakeUserNum)
        tempInteract = np.array(
            [np.random.binomial(1, self.itemP)[self.selectItem] for i in range(self.fakeUserNum)])
        ind = self.interact[userSet, :].nonzero()
        row, col, entries = [], [], []
        for r, c in zip(ind[0].tolist(), ind[1].tolist()):
            if c not in self.selectItem: continue
            c = self.selectItem.index(c)
            row += [r]
            col += [c]
            entries += [self.interact[r, c] * tempInteract[r, c]]
        tempInteract = sp.csr_matrix((entries, (row, col)),
                                     shape=(len(userSet), len(self.selectItem)), dtype=np.float32)
        coo = tempInteract.tocoo()
        inds = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        tempInteract = torch.sparse.FloatTensor(inds, values, coo.shape)
        row, col, entries = [], [], []
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        for step, u in enumerate(self.fakeUser):
            fakeRat = self.G(tempInteract[step].cpu()).detach().cpu().numpy()
            # fakeRat[list(set(list(range(self.itemNum)))-set(self.selectItem))] = -10e8
            fakeRat = self.project(fakeRat, self.maliciousFeedbackNum)
            ind = fakeRat.nonzero()
            self.fakeRat = fakeRat
            for c in ind.tolist():
                c = self.selectItem[c[0]]
                row += [step]
                col += [c]
                entries += [1]
            for c in self.targetItem:
                row += [step]
                col += [c]
                entries += [1]
        fakeRat = csr_matrix((entries, (row, col)), shape=(len(self.fakeUser), self.itemNum), dtype=np.float32)
        return vstack([self.interact, fakeRat])

    def project(self, mat, n):
        matrix = torch.tensor(mat)
        indices = torch.nonzero(matrix > 0.1, as_tuple=True)[0]
        matrix.zero_()
        matrix.scatter_(0, indices, 1)
        return matrix


class Generator(nn.Module):
    """
     In AUSH, Generator is MLP
    """
    def __init__(self, size, layer=2):
        self.layer = layer
        super(Generator, self).__init__()
        self.net = nn.Sequential()
        for i in range(self.layer):
            self.net.add_module('layer_{}'.format(i), nn.Linear(size, size))
            if i != self.layer - 1:
                self.net.add_module('bias_{}'.format(i), nn.ReLU())
            else:
                self.net.add_module('bias_{}'.format(i), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
     In AUSH, Discriminator is Linear layer
    """
    def __init__(self, size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(size, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
