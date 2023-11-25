import numpy as np
import random
import torch
import os
import scipy.sparse as sp
from copy import deepcopy
from util.tool import targetItemSelect
from util.sampler import next_batch_pairwise
from scipy.sparse import vstack, csr_matrix
from util.loss import l2_reg_loss, bpr_loss
from util.algorithm import find_k_largest

class DLAttack():
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

        self.batchSize = 256

    def posionDataAttack(self, recommender):
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        topk = min(recommender.topN)
        p = torch.ones(self.itemNum).cuda()
        sigma = 0.8
        for user in self.fakeUser:
            self.fakeUserInject(recommender,user)
            uiAdj = recommender.data.matrix()
            # outer optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                tmpRecommender.data.user_num + self.itemNum, tmpRecommender.data.user_num + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:tmpRecommender.data.user_num, tmpRecommender.data.user_num:] = uiAdj2
            tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            tmpRecommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=5)
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                with torch.no_grad():
                    Pu, Pi = tmpRecommender.model()
                scores = torch.zeros((uiAdj.shape[0], self.itemNum))
                for batch in range(0,uiAdj.shape[0], self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
                # scores = torch.matmul(Pu, Pi.transpose(0, 1))
                nozeroInd = uiAdj2.indices
                scores[nozeroInd[0],nozeroInd[1]] = -10e8
                # scores = torch.matmul(Pu, Pi.transpose(0, 1))
                # scores = scores - 10e8 * torch.tensor(uiAdj2.todense()).cuda()
                _, top_items = torch.topk(scores, topk)
                top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                for n, batch in enumerate(next_batch_pairwise(self.data, tmpRecommender.args.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    rec_user_emb, rec_item_emb = tmpRecommender.model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
                    users, pos_items, neg_items = [], [], []
                    for idx, u_index in enumerate(list(set(user_idx))):
                        for item in self.targetItem:
                            users.append(u_index)
                            pos_items.append(item)
                            neg_items.append(top_items[u_index][-1])
                    user_emb_cw = Pu[users]
                    pos_items_emb = Pi[pos_items]
                    neg_items_emb = Pi[neg_items]
                    pos_score = torch.mul(user_emb_cw, pos_items_emb).sum(dim=1)
                    neg_score = torch.mul(user_emb_cw, neg_items_emb).sum(dim=1)
                    CWloss = neg_score - pos_score
                    CWloss = CWloss.mean()
                    batch_loss = CWloss + bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(tmpRecommender.args.reg, user_emb,
                                                        pos_item_emb, Pu @ Pi.T)
                    optimizer_attack.zero_grad()
                    batch_loss.backward()
                    optimizer_attack.step()
            with torch.no_grad():
                Pu, Pi = tmpRecommender.model()
            r = Pu[user,:] @ Pi.T
            r = r * p
            m, ind = self.project(r, self.maliciousFeedbackNum)
            uiAdj2[user, :] = m.cpu()
            p[ind] = p[ind] * sigma
            if max(p) < 1:
                p = torch.ones(self.itemNum).cuda()

            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                recommender.data.user_num + self.itemNum, recommender.data.user_num + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:recommender.data.user_num, recommender.data.user_num:] = uiAdj2
            recommender.model._init_uiAdj(ui_adj + ui_adj.T)

            uiAdj = uiAdj2[:, :]
        self.interact = uiAdj
        return self.interact

    def project(self, mat, n):
        matrix = torch.tensor(deepcopy(mat))
        _, indices = torch.topk(matrix, n, dim=0)
        matrix.zero_()
        matrix.scatter_(0, indices, 1)
        return matrix, indices

    def fakeUserInject(self, recommender, user):
        Pu, Pi = recommender.model()
        recommender.data.user_num += 1
        recommender.data.user["fakeuser{}".format(recommender.data.user_num)] = len(recommender.data.user)
        recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(recommender.data.user_num)

        row, col, entries = [], [], []
        for i in self.targetItem:
            recommender.data.training_data.append((recommender.data.id2user[user], recommender.data.id2item[i]))
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


