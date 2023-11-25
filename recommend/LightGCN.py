import torch
import torch.nn as nn
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss
import time
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from os.path import abspath
import sys
from util.metrics import ranking_evaluation
from util.FileIO import FileIO
from util.logger import Log
import scipy.sparse as sp
import numpy as np


class LightGCN():
    def __init__(self, args, data):
        print("Recommender: LightGCN")
        self.data = data
        self.args = args
        self.bestPerformance = []
        self.recOutput = []
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model = LGCN_Encoder(self.data, self.args.emb_size, self.args.n_layers)

    def train(self, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0, optimizer=None, evalNum=5):
        self.bestPerformance=[]
        model = self.model.cuda()
        if optimizer is None: 
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        else:
            self.optimizer = optimizer
        if requires_embgrad:
            model.requires_grad = True
            self.usergrad = torch.zeros((self.data.user_num, self.args.emb_size)).cuda()
            self.itemgrad = torch.zeros((self.data.item_num, self.args.emb_size)).cuda()
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
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.args.reg, user_emb,
                                                                                          pos_item_emb)
                self.optimizer.zero_grad()
                batch_loss.backward()

                if requires_adjgrad and maxEpoch - epoch < gradIterationNum:
                    self.Matgrad += self.model.sparse_norm_adj.grad
                elif requires_embgrad and maxEpoch - epoch < gradIterationNum:
                    self.usergrad += self.model.embedding_dict["user_emb"].grad
                    self.itemgrad += self.model.embedding_dict["item_emb"].grad

                self.optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % evalNum == 0:
                self.evaluate(epoch)
                # self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if requires_adjgrad and requires_embgrad:
            return (self.Matgrad + self.Matgrad.T)[:self.data.user_num, self.data.user_num:], \
                   self.user_emb, self.item_emb, self.usergrad, self.itemgrad
        elif requires_adjgrad:
            return (self.Matgrad + self.Matgrad.T)[:self.data.user_num, self.data.user_num:]
        elif requires_embgrad:
            return self.user_emb, self.item_emb, self.usergrad, self.itemgrad

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def evaluate(self, epoch):
        print('Evaluating the model...')
        rec_list, _ = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list,ranking_evaluation(self.data.test_set, rec_list, self.topN)

    # def test(self):
    #     def process_bar(num, total):
    #         rate = float(num) / total
    #         ratenum = int(50 * rate)
    #         r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum * 2)
    #         sys.stdout.write(r)
    #         sys.stdout.flush()

    #     # predict
    #     rec_list = {}
    #     user_count = len(self.data.test_set)
    #     for i, user in enumerate(self.data.test_set):
    #         candidates = self.predict(user)
    #         rated_list, li = self.data.user_rated(user)
    #         for item in rated_list:
    #             candidates[self.data.item[item]] = -10e8
    #         ids, scores = find_k_largest(self.max_N, candidates)
    #         item_names = [self.data.id2item[iid] for iid in ids]
    #         rec_list[user] = list(zip(item_names, scores))
    #         if i % 1000 == 0:
    #             process_bar(i, user_count)
    #     process_bar(user_count, user_count)
    #     print('')

    #     self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
    #     for user in self.data.test_set:
    #         line = user + ':'
    #         for item in rec_list[user]:
    #             line += ' (' + item[0] + ',' + str(item[1]) + ')'
    #             if item[0] in self.data.test_set[user]:
    #                 line += '*'
    #         line += '\n'
    #         self.recOutput.append(line)
    #     # output prediction result
    #     self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
    #     print('The result of %s:\n%s' % (self.args.model_name, ''.join(self.result)))
    #     return self.result


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_uiAdj(self, ui_adj):
        self.sparse_norm_adj = sp.diags(np.array((1 / np.sqrt(ui_adj.sum(1)))).flatten()) @ ui_adj @ sp.diags(
            np.array((1 / np.sqrt(ui_adj.sum(0)))).flatten())
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.sparse_norm_adj).cuda()

    def attack_emb(self, users_emb_grad, items_emb_grad):
        with torch.no_grad():
            self.embedding_dict['user_emb'] += users_emb_grad
            self.embedding_dict['item_emb'] += items_emb_grad

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
