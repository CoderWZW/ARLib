import numpy as np
import random
import torch
from util.tool import targetItemSelect
from util.algorithm import find_k_largest
import scipy.sparse as sp
from copy import deepcopy

class FedRecAttack():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.data = data
        self.interact = data.matrix()
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.selectSize = arg.selectSize
        self.userNum = data.user_num
        self.itemNum = data.item_num
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackSize = (self.interact.sum() / data.user_num) / data.item_num
            self.selectSize = self.maliciousFeedbackSize / 2
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        self.gradMaxLimitation = int(arg.gradMaxLimitation)
        self.gradNumLimitation = int(arg.gradNumLimitation)
        self.BiLevelOptimizationEpoch = 5
        self.attackEpoch = int(arg.attackEpoch)

    def project(self, mat, n):
        matrix = mat[:, :]
        for i in range(matrix.shape[0]):
            subMatrix = torch.tensor(matrix[i, :].todense())
            v = torch.topk(subMatrix.flatten(), n)[0][-1]
            subMatrix[subMatrix >= v] = 1
            subMatrix[subMatrix < v] = 0
            if subMatrix.sum() > n:
                ind = (subMatrix == 1).nonzero()
                k = 0
                while subMatrix.sum() > n:
                    subMatrix[0, ind[k][1]] = 0
                    k += 1
            matrix[i, :] = subMatrix[:, :].flatten()
        return matrix

    def dataUpdate(self, recommender):
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)
        n_nodes = recommender.data.user_num + recommender.data.item_num
        row_idx = [recommender.data.user[pair[0]] for pair in recommender.data.training_data]
        col_idx = [recommender.data.item[pair[1]] for pair in recommender.data.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + recommender.data.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        recommender.data.ui_adj = adj_mat
        recommender.data.norm_adj = recommender.data.normalize_graph_mat(recommender.data.ui_adj)
        row, col, entries = [], [], []
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]
        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)
    def posionDataAttack(self, recommender):
        baseUiAdj = self.data.matrix()
        Pu, Pi = recommender.model()
        baseRecGraph = torch.zeros((Pu.shape[0], Pi.shape[0]))
        for u in range(Pu.shape[0]):
            itemScores = Pu[u, :] @ Pi.T
            itemScores[baseUiAdj[u, :].nonzero()[1]] = -10e8
            baseRecGraph[u, find_k_largest(20, itemScores.tolist())[0]] = 1
        maxRecNumItemInd = baseRecGraph.sum(0).argsort(descending=True)[:int(Pi.shape[0] / 10)].tolist()

        optimizer = torch.optim.SGD(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        self.dataUpdate(recommender)
        recommender.__init__(recommender.args, recommender.data)
        newAdj = recommender.data.matrix()
        with torch.no_grad():
            recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
            recommender.model.embedding_dict['item_emb'][:] = Pi
        self.controlledUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        recommender.train(Epoch=self.attackEpoch, optimizer=optimizer, evalNum=5)
        originRecommender = deepcopy(recommender)
        uiAdj = newAdj[:, :]
        for u in self.controlledUser:
            uiAdj[u, :] = 0
            uiAdj[u, self.targetItem] = 1
            uiAdj[u, maxRecNumItemInd[:int(self.maliciousFeedbackSize * self.itemNum)]] = 1

        recommender = deepcopy(originRecommender)
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        for epoch in range(self.BiLevelOptimizationEpoch):
            # outer optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                            dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj
            recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            recommender.train(Epoch=self.attackEpoch, optimizer=optimizer, evalNum=3)

            # inner optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            for u in self.controlledUser:
                for i in range(self.itemNum):
                    if uiAdj2[u, i] == 0:
                        uiAdj2[u, i] = 10e-8

            for _ in range(5):
                ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                       dtype=np.float32)
                ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj
                tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
                tmpRecommender.model.sparse_norm_adj.requires_grad = True

                user_embed, item_embed = tmpRecommender.model()
                
                scores = torch.matmul(user_embed, item_embed.transpose(0, 1))
                _, top_items = torch.topk(scores, 50)

                top_items = [[iid.item() for iid in user_top] for user_top in top_items]

                # Optimize CW loss
                users, pos_items, neg_items = [], [], []
                for idx, u_index in enumerate(list(range(self.userNum))):
                    for item in self.targetItem:
                        users.append(u_index)
                        pos_items.append(item)
                        neg_items.append(top_items[u_index].pop())

                user_emb = user_embed[users]
                pos_items_emb = item_embed[pos_items]
                neg_items_emb = item_embed[neg_items]
                pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
                neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
                loss = neg_score - pos_score
                loss = loss.mean()

                grad = torch.autograd.grad(loss, tmpRecommender.model.sparse_norm_adj)[0].to_dense()[
                       :self.userNum + self.fakeUserNum,
                       self.userNum + self.fakeUserNum:][self.controlledUser, :]
                self.g = grad
                with torch.no_grad():
                    for no, u in enumerate(self.controlledUser):
                        for step, item in enumerate(maxRecNumItemInd):
                            # uiAdj2[u, item] -= 0.1 * grad[no, step].item() * np.sqrt(uiAdj2[u, :].sum()) * np.sqrt(
                            #     uiAdj2[:, item].sum())
                            uiAdj2[u, item] -= 0.1 * grad[no,step].item() *(2*np.power(uiAdj2[:, item].sum(),-1.5)*(uiAdj2[u, item]/np.sqrt(uiAdj2[u, :].sum()))+
                                    2*np.power(uiAdj2[u, :].sum(),-1.5)*(uiAdj2[u, item]/np.sqrt(uiAdj2[:, item].sum()))+
                                    (np.sqrt(uiAdj2[u, :].sum()) * np.sqrt(uiAdj2[:, item].sum()))**-1)
                            if uiAdj2[u, item] > 1:
                                uiAdj2[u, item] = 1
                            elif uiAdj2[u, item] < 0:
                                uiAdj2[u, item] = 0
                
            uiAdj2[self.controlledUser, :] = self.project(uiAdj2[self.controlledUser, :],
                                                          int(self.maliciousFeedbackSize * self.itemNum))
            for u in self.controlledUser:
                for i in self.targetItem:
                    uiAdj2[u, i] = 1
            uiAdj = uiAdj2[:, :]
            print("attack step {} is over\n".format(epoch + 1))
        self.interact = uiAdj
        return self.interact

    def gradientattack(self, recommender):
        self.controlledUser = random.sample(set(range(self.data.user_num)), self.fakeUserNum)
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        for i in range(self.BiLevelOptimizationEpoch):
            user_embed, item_embed, usergrad, itemgrad = recommender.train(requires_embgrad=True, Epoch=self.attackEpoch,optimizer=optimizer, evalNum=4)
            user_embed.requires_grad = True
            item_embed.requires_grad = True
            scores = torch.matmul(user_embed, item_embed.transpose(0, 1))
            _, top_items = torch.topk(scores, 50)

            top_items = [[iid.item() for iid in user_top] for user_top in top_items]

            # Optimize CW loss
            users, pos_items, neg_items = [], [], []
            for idx, u_index in enumerate(self.controlledUser):
                user = self.data.id2user[u_index]
                for item in self.targetItem:
                    if self.data.id2item[item] not in self.data.training_set_u[user]:
                        users.append(u_index)
                        pos_items.append(item)
                        neg_items.append(top_items[u_index].pop())

            user_emb = user_embed[users]
            pos_items_emb = item_embed[pos_items]
            neg_items_emb = item_embed[neg_items]
            pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
            neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
            loss = neg_score - pos_score

            # loss has limitation, if x>0, x=x; else x <0, loss = e^x-1
            loss[loss < 0] = torch.exp(loss[loss < 0]) - 1
            loss = loss.sum()
            loss.backward(retain_graph=True)

            user_embed_grad = user_embed.grad
            item_embed_grad = item_embed.grad

            # grad has two limitations,
            # 1. the number of grad change can not be more than grad_num_limitation
            item_grad_indexs = random.sample(set(range(self.data.item_num)),
                                             self.gradNumLimitation - len(self.targetItem))
            item_grad_indexs = item_grad_indexs + self.targetItem

            new_item_grad = torch.zeros_like(item_embed_grad)
            new_item_grad[item_grad_indexs] = item_embed_grad[item_grad_indexs]

            # 2. grad can not be more than grad_max_limitation
            items_embed_grad_norm = new_item_grad.norm(2, dim=-1, keepdim=True)
            grad_max = self.gradMaxLimitation
            too_large = items_embed_grad_norm[:, 0] > grad_max
            new_item_grad[too_large] /= (items_embed_grad_norm[too_large] / grad_max)

            recommender.model.attack_emb(user_embed_grad, new_item_grad)