import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect,getPopularItemId
from util.metrics import AttackMetric
from util.algorithm import find_k_largest
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from util.loss import bpr_loss, l2_reg_loss
from sklearn.neighbors import LocalOutlierFactor as LOF
from util.sampler import next_batch_pairwise
from recommend.LightGCN import LightGCN
import logging


class GTA():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]

        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # limitation 
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.batchSize = 128


    def posionDataAttack(self,recommend):
        recommender = proxyLG(recommend.args, recommend.data, self.targetItem)
        self.fakeUserInject(recommender)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate)
        recommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=5)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
        seedItem = random.sample(getPopularItemId(recommend.data.matrix(),self.itemNum//5).tolist()[0],self.maliciousFeedbackNum//2)
        for epoch in range(self.Epoch):
            # inner optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj

            recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            recommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=5)
            attackmetrics = AttackMetric(recommender, self.targetItem, [topk])
            targetHitRate = attackmetrics.hitRate()[0]
            print(targetHitRate)
            if targetHitRate > bestTargetHitRate:
                bestAdj = uiAdj[:,:]
                bestTargetHitRate = targetHitRate
            uiAdj = bestAdj[:,:]
            uiAdj2 = uiAdj[:, :]
            Pu, Pi = recommender.model()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            for u in self.fakeUser:
                uiAdj2[u, seedItem] = 0
            uiAdj2[self.fakeUser, :] = self.project(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum//2)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem + seedItem] = 1
            uiAdj = uiAdj2
            print("BiLevel epoch {} is over\n".format(epoch + 1))
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

    def fakeUserInject(self, recommender):
        recommender.model = recommender.model.cuda()
        Pu, Pi = recommender.model()
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)

        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem =  random.sample(set(list(range(self.itemNum))),self.maliciousFeedbackNum)
            for i in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u],recommender.data.id2item[i]))
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]

        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)

        recommender.__init__(recommender.args, recommender.data, self.targetItem)
        # recommender.model = recommender.model.cuda()
        ui_adj = sp.csr_matrix(([], ([], [])), shape=(
            self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                dtype=np.float32)
        ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = recommender.data.matrix()
        recommender.model._init_uiAdj(ui_adj + ui_adj.T)
        recommender.train(Epoch=30)

class proxyLG(LightGCN):
    def __init__(self, args, data, targetItem):
        super(proxyLG, self).__init__(args, data)
        self.userNum = data.user_num
        self.itemNum = data.item_num
        self.targetItem = targetItem
        self.batchSize = 1024
    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None, evalNum=5):
        self.bestPerformance = []
        model = self.model.cuda()
        uiAdj2 = self.data.matrix()
        topk = min(self.topN)
        if optimizer is None: optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        if requires_embgrad:
            model.requires_grad = True
            self.usergrad = torch.zeros_like(
                torch.cat([self.model.embedding_dict['user_mf_emb'], self.model.embedding_dict['user_mlp_emb']],
                          1)).cuda()
            self.itemgrad = torch.zeros_like(
                torch.cat([self.model.embedding_dict['item_mf_emb'], self.model.embedding_dict['item_mlp_emb']],
                          1)).cuda()
        elif requires_adjgrad:
            self.model.sparse_norm_adj.requires_grad = True
            self.Matgrad = torch.zeros(
                (self.data.user_num + self.data.item_num, self.data.user_num + self.data.item_num)).cuda()
        maxEpoch = self.args.maxEpoch
        if Epoch: maxEpoch = Epoch
        for epoch in range(maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb = model()
                user_emb1, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]

                Pu = rec_user_emb
                Pi = rec_item_emb
                scores = torch.zeros((self.userNum, self.itemNum))
                for batch in range(0,self.userNum, self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
                # scores = torch.matmul(Pu, Pi.transpose(0, 1))
                nozeroInd = uiAdj2.indices
                scores[nozeroInd[0],nozeroInd[1]] = -10e8
                _, top_items = torch.topk(scores, topk)
                top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                users, pos_items, neg_items = [], [], []
                for idx, u_index in enumerate(list(range(self.userNum))):
                    for item in self.targetItem:
                        users.append(u_index)
                        pos_items.append(item)
                        neg_items.append(top_items[u_index].pop())
                user_emb = Pu[users]
                pos_items_emb = Pi[pos_items]
                neg_items_emb = Pi[neg_items]
                pos_score = torch.mul(user_emb, pos_items_emb).mean(dim=1)
                neg_score = torch.mul(user_emb, neg_items_emb).mean(dim=1)
                CWloss = neg_score - pos_score
                CWloss = CWloss.mean()

                batch_loss = 0.01*CWloss + bpr_loss(user_emb1, pos_item_emb, neg_item_emb) + l2_reg_loss(self.args.reg, user_emb1,
                                                                                          pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()

                if requires_adjgrad and maxEpoch - epoch < gradIterationNum:
                    self.Matgrad += self.model.sparse_norm_adj.grad
                elif requires_embgrad and maxEpoch - epoch < gradIterationNum:
                    self.usergrad += torch.cat(
                        [self.model.embedding_dict['user_mf_emb'].grad, self.model.embedding_dict['user_mlp_emb'].grad],
                        1)
                    self.itemgrad += torch.cat(
                        [self.model.embedding_dict['item_mf_emb'].grad, self.model.embedding_dict['item_mlp_emb'].grad],
                        1)

                optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % evalNum == 0:
                self.evaluate(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if requires_adjgrad and requires_embgrad:
            return (self.Matgrad + self.Matgrad.T)[:self.data.user_num, self.data.user_num:], \
                   self.user_emb, self.item_emb, self.usergrad, self.itemgrad
        elif requires_adjgrad:
            return (self.Matgrad + self.Matgrad.T)[:self.data.user_num, self.data.user_num:]
        elif requires_embgrad:
            return self.user_emb, self.item_emb, self.usergrad, self.itemgrad
