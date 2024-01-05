import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect
from util.metrics import AttackMetric
from util.algorithm import find_k_largest
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from util.loss import bpr_loss, l2_reg_loss
from sklearn.neighbors import LocalOutlierFactor as LOF
from recommender.GMF import GMF
import logging


class CLeaR():
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

        self.batchSize = 2048


    def posionDataAttack(self, recommender):
        self.fakeUserInject(recommender)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate/10)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
        ind = None
        for epoch in range(self.Epoch):
            # outer optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj2
            tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                Pu, Pi = tmpRecommender.model()
                scores = torch.zeros((self.userNum + self.fakeUserNum, self.itemNum))
                for batch in range(0,self.userNum + self.fakeUserNum, self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
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
                pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
                neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
                CWloss = neg_score - pos_score
                CWloss = CWloss.mean()
                # CWloss = - pos_score
                # CWloss = CWloss.mean()                
                def spectral_feature_augmentation(H, k):
                    # Initialize r(0) from a normal distribution
                    r = torch.randn(H.size(1))

                    # Ensure the computations are on the same device as H
                    r = r.to(H.device)

                    # Iterate k times
                    for i in range(k):
                        r = H.T @ H @ r

                    # Calculate the augmented feature map
                    H_aug = H - (H @ torch.outer(r, r)) / torch.norm(r)**2

                    return H_aug
                # item_emb_cat = torch.cat((pos_items_emb, neg_items_emb), dim=0)
                # item_SFA = spectral_feature_augmentation(item_emb_cat, 1)
                # import torch.nn.functional as F
                # sfaloss1 = F.l1_loss(item_SFA, item_emb_cat)
                # user_SFA = spectral_feature_augmentation(user_emb, 1)
                # import torch.nn.functional as F
                # sfaloss2 = F.l1_loss(user_SFA, user_emb)

                item_emb_cat = torch.cat((pos_items_emb, neg_items_emb), dim=0)
                emb_cat = torch.cat((user_emb, item_emb_cat), dim=0)
                SFA = spectral_feature_augmentation(emb_cat, 1)
                import torch.nn.functional as F
                sfaloss = F.l1_loss(SFA, emb_cat)
                lossall = CWloss + sfaloss
                optimizer_attack.zero_grad()
                lossall.backward()
                optimizer_attack.step()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            uiAdj2[self.fakeUser, :], _ = self.project(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem] = 1

            uiAdj = uiAdj2[:, :]

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
        return matrix,indices

    def fakeUserInject(self, recommender):
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

        recommender.__init__(recommender.args, recommender.data)
        with torch.no_grad():
            try:
                recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
                recommender.model.embedding_dict['item_emb'][:] = Pi
            except:
                recommender.model.embedding_dict['user_mf_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], :Pu.shape[1]//2]
                recommender.model.embedding_dict['user_mlp_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], Pu.shape[1]//2:]
                recommender.model.embedding_dict['item_mf_emb'][:] = Pi[:, :Pi.shape[1]//2]
                recommender.model.embedding_dict['item_mlp_emb'][:] = Pi[:, Pi.shape[1]//2:]

        recommender.model = recommender.model.cuda()
