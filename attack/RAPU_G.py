import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from util.tool import targetItemSelect
import math
import time
from scipy import sparse
from scipy.sparse import vstack, csr_matrix


class RAPU_G():
    def __init__(self, arg, data):
        """
         :param arg: parameter configuration
         :param data: dataLoder
         """
        self.interact = data.matrix()
        self.data = data
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.selectSize = arg.selectSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackSize = (self.interact.sum() / self.userNum) / self.itemNum
            self.selectSize = self.maliciousFeedbackSize / 2
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False
        self.BlackBoxAdvTrainer = BlackBoxAdvTrainer(self.userNum, self.itemNum, self.fakeUserNum, self.targetItem,
                                                     self.maliciousFeedbackSize)

    def posionDataAttack(self):
        userId, itemId, rating = [], [], []
        for i in range(self.userNum):
            for j in range(self.itemNum):
                if self.interact[i,j] != 0:
                    userId.append(i)
                    itemId.append(j)
                    rating.append(1)
        train_data, val_data, test_data = split_train_test(self.userNum, self.itemNum, userId,
                                                           itemId, rating)
        fakeRat = self.BlackBoxAdvTrainer.fit(train_data=train_data, test_data=test_data)
        print("over")
        return vstack([self.interact, csr_matrix(fakeRat)])


class WMF(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(WMF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()
        self.device = torch.device("cuda")
        self.batch_size = 2048
        self.weight_alpha = 20
        self.sigma = 1
        self.b = 1
        self.barr = 0.5
        self.EM_epoch = 3

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.latent_size))),
        })
        return embedding_dict

    def fit_adj(self, data_tensor, epoch_num, unroll_steps, n_fakes, target_items):
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=2e-2,
                                          weight_decay=1e-5)
        import higher
        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)

        model = self.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.batch_size
                      if self.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(idx_list, batch_size=batch_size):
                Pu, Pi = model()
                logits = Pu[batch_idx] @ Pi.T
                loss = mse_loss(data=data_tensor[batch_idx],
                                logits=logits,
                                weight=self.weight_alpha).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(idx_list, batch_size=batch_size):
                    fake_user_idx = batch_idx[batch_idx >= (n_rows - n_fakes)]
                    Pu, Pi = fmodel()
                    fake_logits = Pu[fake_user_idx] @ Pi.T
                    fake_user_loss = mse_loss(data=data_tensor[fake_user_idx],
                                              logits=fake_logits,
                                              weight=self.weight_alpha).sum()

                    normal_user_idx = batch_idx[batch_idx < (n_rows - n_fakes)]
                    Pu, Pi = fmodel()
                    normal_logits = Pu[normal_user_idx] @ Pi.T
                    if i <= epoch_num - self.EM_epoch:
                        normal_user_loss = mse_loss(
                            data=data_tensor[normal_user_idx],
                            logits=normal_logits,
                            weight=self.weight_alpha).sum()
                    else:
                        print(".", end='')
                        normal_user_loss = pgm_loss(
                            data=data_tensor[normal_user_idx],
                            logits=normal_logits,
                            weight=self.weight_alpha,
                            sigma=self.sigma,
                            bar_r=self.barr).sum()

                    loss = fake_user_loss + normal_user_loss
                    epoch_loss += loss.item()
                    diffopt.step(loss)

            fmodel.eval()

            Pu, Pi = fmodel()
            predictions = Pu @ Pi.T
            predictions_ranked, ranking_ind = torch.topk(predictions, 50)
            logits_topK = predictions_ranked[:-n_fakes, ]
            logits_target_items = predictions[:-n_fakes, target_items]
            adv_loss = WMW_loss_sigmoid(
                logits_topK=logits_topK,
                logits_target_items=logits_target_items,
                offset=self.b)

            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            model.load_state_dict(fmodel.state_dict())

            return adv_loss.item(), adv_grads[-n_fakes:, ]

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']

    def validate(self, train_data, test_data, train_epoch, target_items):
        normal_user_num = test_data.shape[0]

        recommendations = self.recommend(train_data, top_k=100)

        k = 50
        hit_num = 0
        targets = target_items
        topK_rec = recommendations[:normal_user_num, :k]
        for i in range(normal_user_num):
            inter_set_len = len(set(topK_rec[i]).intersection(set(targets)))
            if inter_set_len > 0:
                hit_num += 1
        hit_ratio = hit_num / normal_user_num

        result = {'TargetHR@50': hit_ratio}

        return result

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        model = self.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(idx_list,
                                       batch_size=512):
                batch_data = data[batch_idx].toarray()
                Pu, Pi = model()
                preds = Pu[batch_idx] @ Pi.T
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations


class BlackBoxAdvTrainer:
    def __init__(self, n_users, n_items, n_fakes, target_items, maliciousFeedbackSize):

        self.n_users = n_users
        self.n_items = n_items
        self.n_fakes = n_fakes
        self.target_items = target_items
        self.golden_metric = "TargetHR@50"
        self.device = torch.device("cuda")
        self.adv_epochs = 30
        self.unroll_steps = 20
        self.EM = 1
        self.sigma = 0.3
        self.b = 0.1
        self.proj_topk = maliciousFeedbackSize
        self.p_item = int(n_items * maliciousFeedbackSize)

    def _initialize(self, train_data):
        fake_data = self.init_fake_data(train_data=train_data)

        self.fake_tensor = sparse2tensor(fake_data)
        self.fake_tensor.requires_grad_()

        self.optimizer = torch.optim.SGD([self.fake_tensor],
                                         lr=1,
                                         momentum=0.95)

    def train_epoch(self, train_data, epoch_num):
        def compute_adv_grads():
            model = WMF(self.n_users + self.n_fakes, self.n_items, 32)

            data_tensor = torch.cat([
                sparse2tensor(train_data).to(self.device),
                self.fake_tensor.detach().clone().to(self.device)
            ], dim=0)
            self.test2 = train_data
            self.test3 = self.fake_tensor
            data_tensor.requires_grad_()
            epoch = 50
            adv_loss_, adv_grads_ = model.fit_adj(data_tensor, epoch, self.unroll_steps, self.n_fakes,
                                                  self.target_items)
            return model, adv_loss_, adv_grads_

        def project_tensor(fake_tensor):
            upper = self.p_item
            values, idxs = torch.topk(fake_tensor, upper, dim=1)
            user_idxs = torch.tensor(range(fake_tensor.shape[0])).reshape(
                -1, 1).repeat(1, upper).reshape(-1)
            item_idxs = idxs.reshape(-1)
            new_fake_tensor = torch.zeros_like(fake_tensor)
            new_fake_tensor[user_idxs, item_idxs] = 1
            return new_fake_tensor

        sur_trainer = None
        new_fake_tensor = None
        t1 = time.time()
        self.optimizer.zero_grad()

        sur_trainer, adv_loss, adv_grads = compute_adv_grads()
        adv_grads[:, self.target_items] = 0.0
        print(
            "\nAdversarial training [{:.1f} s],  epoch: {}, loss: {:.4f}".
                format(time.time() - t1, epoch_num, adv_loss),
            end='\t\t')

        normalized_adv_grads = adv_grads / adv_grads.norm(
            p=2, dim=1, keepdim=True)
        if self.fake_tensor.grad is None:
            self.fake_tensor.grad = normalized_adv_grads.detach().cpu()
        else:
            self.fake_tensor.grad.data = normalized_adv_grads.detach().cpu(
            )

        self.optimizer.step()

        new_fake_tensor = project_tensor(fake_tensor=self.fake_tensor.data.clone(), )

        new_fake_tensor[:, self.target_items] = 1.0

        return sur_trainer, new_fake_tensor

    def evaluate_epoch(self, trainer, train_data, test_data):
        result = trainer.validate(train_data=train_data,
                                  test_data=test_data,
                                  train_epoch=-1,
                                  target_items=self.target_items)
        return result

    def fit(self, train_data, test_data):
        self._initialize(train_data)
        best_fake_data, best_perf = None, 0.0
        cur_fake_tensor = self.fake_tensor.detach().clone()
        for epoch_num in range(1, self.adv_epochs + 1):
            cur_sur_trainer, new_fake_tensor = self.train_epoch(
                train_data, epoch_num)
            print("Total changes: {}".format(
                (new_fake_tensor - cur_fake_tensor).abs().sum().item()),
                end='\t\t')

            self.fake_tensor.data = new_fake_tensor.detach().clone()
            cur_fake_tensor = new_fake_tensor.detach().clone()

            cur_fake_data = tensor2sparse(cur_fake_tensor)
            result = self.evaluate_epoch(trainer=cur_sur_trainer,
                                         train_data=stack_csrdata(
                                             train_data, cur_fake_data),
                                         test_data=test_data)

            cur_perf = result[self.golden_metric]
            print("Hit@50:%.2f" % cur_perf, end='\t\t')
            if cur_perf > best_perf:
                print("Better fake data with "
                      "{}={:.4f}".format(self.golden_metric, cur_perf),
                      end='\t\t')
                best_fake_data, best_perf = cur_fake_data, cur_perf
        self.best_fake_data = best_fake_data
        return self.best_fake_data

    def init_fake_data(self, train_data):
        train_data = train_data.toarray()
        max_allowed_click = 100
        user_clicks = train_data.sum(1)
        qual_users = np.where(user_clicks <= max_allowed_click)[0]

        indices = np.arange(len(qual_users))
        np.random.shuffle(indices)
        sampled_users = qual_users[:self.n_fakes]
        fake_data = sparse.csr_matrix(train_data[sampled_users],
                                      dtype=np.float64,
                                      shape=(self.n_fakes, self.n_items))
        return fake_data


def sparse2tensor(sparse_data):
    return torch.Tensor(sparse_data.toarray())


def tensor2sparse(tensor):
    return sparse.csr_matrix(tensor.detach().cpu().numpy())


def stack_csrdata(data1, data2):
    return sparse.vstack((data1, data2), format="csr")


def mse_loss(data, logits, weight):
    weights = torch.ones_like(data)
    weights[data > 0] = weight
    res = weights * (data - logits) ** 2
    return res.sum(1)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def pgm_loss(data, logits, weight, sigma, bar_r=0.1):
    gamma_u, iota_i, eta_u, zeta_i, delta_p_ui, psi_p_ui = EM_opt(
        data,
        logits.clone().detach(), sigma, bar_r)

    delta_1_psi_1 = torch.mul(delta_p_ui, psi_p_ui)
    delta_1_psi_1[torch.isnan(delta_1_psi_1)] = 0
    mask_ratio_delta_1_psi_1 = float(
        torch.sum(delta_1_psi_1).item() /
        (delta_1_psi_1.shape[0] * delta_1_psi_1.shape[1]))

    weights = torch.ones_like(data)
    weights[data > 0] = weight
    mse = weights * (data - logits) ** 2
    res = delta_1_psi_1 * mse / mask_ratio_delta_1_psi_1

    return res.sum(1)


def EM_opt(data, logits, sigma, bar_r):
    PI = np.pi
    bar_sigma = sigma
    num_u, num_i = data.shape
    gamma_u = torch.full((num_u, 1), 0.8, dtype=float).cuda()
    iota_i = torch.full((1, num_i), 0.8, dtype=float).cuda()
    eta_u = torch.full((num_u, 1), 0.8, dtype=float).cuda()
    zeta_i = torch.full((1, num_i), 0.8, dtype=float).cuda()

    hat_normal_0_exp = torch.exp(-torch.div(logits ** 2, 2 * sigma ** 2))
    hat_normal_PDF_0 = torch.div(hat_normal_0_exp, (math.sqrt(2 * PI) * sigma))
    bar_normal_0_exp = torch.exp(-torch.div(bar_r ** 2, 2 * bar_sigma ** 2))
    bar_normal_PDF_0 = torch.div(bar_normal_0_exp,
                                 (math.sqrt(2 * PI) * bar_sigma))

    hat_normal_1_exp = torch.exp(-torch.div((logits - 1) ** 2, 2 * sigma ** 2))
    hat_normal_PDF_1 = torch.div(hat_normal_1_exp, (math.sqrt(2 * PI) * sigma))
    bar_normal_1_exp = torch.exp(-torch.div((bar_r - 1) ** 2, 2 * bar_sigma ** 2))
    bar_normal_PDF_1 = torch.div(bar_normal_1_exp,
                                 (math.sqrt(2 * PI) * bar_sigma))

    delta_p_ui = None
    psi_p_ui = None
    while True:
        delta_eq_1 = torch.div(
            gamma_u.repeat(1, num_i) + iota_i.repeat(num_u, 1), 2.0)
        psi_eq_1 = torch.div(
            eta_u.repeat(1, num_i) + zeta_i.repeat(num_u, 1), 2.0)

        delta_exp_r0 = torch.mul(psi_eq_1, hat_normal_PDF_0) + torch.mul(
            (1 - psi_eq_1), bar_normal_PDF_0)

        delta_numerator = torch.mul(delta_eq_1, delta_exp_r0)
        delta_denominator = delta_numerator + (1 - delta_eq_1)
        delta_p_ui = torch.div(delta_numerator, delta_denominator)
        delta_p_ui[data > 0] = 1.0

        delta_p_ui[delta_p_ui >= 0.5] = 1
        delta_p_ui[delta_p_ui < 0.5] = 0

        psi_exp_delta1_r0 = torch.div(torch.mul(psi_eq_1, hat_normal_PDF_0),
                                      delta_exp_r0)

        delta_exp_r1 = torch.mul(psi_eq_1, hat_normal_PDF_1) + torch.mul(
            (1 - psi_eq_1), bar_normal_PDF_1)
        psi_exp_delta1_r1 = torch.div(torch.mul(psi_eq_1, hat_normal_PDF_1),
                                      delta_exp_r1)
        psi_p_ui = torch.where(data > 0, psi_exp_delta1_r1, psi_exp_delta1_r0)
        psi_p_ui[delta_p_ui < 0.5] = np.nan

        new_gamma = torch.mean(delta_p_ui, dim=1).reshape(-1, 1)
        new_iota = torch.mean(delta_p_ui, dim=0).reshape(1, -1)

        new_eta = torch.from_numpy(np.nanmean(psi_p_ui.cpu().numpy(),
                                              axis=1)).reshape(-1, 1).cuda()
        new_zeta = torch.from_numpy(np.nanmean(psi_p_ui.cpu().numpy(),
                                               axis=0)).reshape(1, -1).cuda()

        if is_converge(new_gamma, gamma_u) and is_converge(
                new_iota, iota_i) and is_converge(
            new_eta, eta_u) and is_converge(new_zeta, zeta_i):
            gamma_u = new_gamma
            iota_i = new_iota
            eta_u = new_eta
            zeta_i = new_zeta
            psi_p_ui[psi_p_ui >= 0.5] = 1
            psi_p_ui[psi_p_ui < 0.5] = 0
            break
        gamma_u = new_gamma
        iota_i = new_iota
        eta_u = new_eta
        zeta_i = new_zeta
    return gamma_u, iota_i, eta_u, zeta_i, delta_p_ui, psi_p_ui


def is_converge(new_para, para, rtol=1e-05, atol=1e-08):
    return torch.allclose(new_para, para, rtol=rtol, atol=atol)


def WMW_loss_sigmoid(logits_topK, logits_target_items, offset=0.01):
    loss = 0
    for i in range(logits_target_items.shape[1]):
        cur_target_logits = logits_target_items[:, i].reshape(-1, 1)
        x = (logits_topK - cur_target_logits) / offset
        g = torch.sigmoid(x)
        loss += torch.sum(g)
    return loss


def split_train_test(n_users, n_items, userId, itemId, rating, split_prop=[0.8, 0, 0.2]):
    inter = np.concatenate((np.expand_dims(userId, 1), np.expand_dims(
        itemId, 1), np.expand_dims(rating, 1)),
                           axis=1)
    np.random.shuffle(inter)
    inter_num = inter.shape[0]

    train_inter = inter[:int(inter_num * split_prop[0]), :]
    val_inter = inter[int(inter_num * split_prop[0]):int(inter_num *
                                                         (split_prop[0] +
                                                          split_prop[1])), :]
    test_inter = inter[int(inter_num * (split_prop[0] + split_prop[1])):, :]

    train_data = sparse.csr_matrix((np.ones_like(train_inter[:, 2]),
                                    (train_inter[:, 0], train_inter[:, 1])),
                                   dtype='float64',
                                   shape=(n_users, n_items))
    val_data = sparse.csr_matrix(
        (np.ones_like(val_inter[:, 2]), (val_inter[:, 0], val_inter[:, 1])),
        dtype='float64',
        shape=(n_users, n_items))
    test_data = sparse.csr_matrix(
        (np.ones_like(test_inter[:, 2]), (test_inter[:, 0], test_inter[:, 1])),
        dtype='float64',
        shape=(n_users, n_items))

    return train_data, val_data, test_data
