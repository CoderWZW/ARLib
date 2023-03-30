import numpy as np
import random
import torch
from util.tool import targetItemSelect


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

        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackSize = (self.interact.sum() / data.user_num) / data.item_num
            self.selectSize = self.maliciousFeedbackSize / 2
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]

        self.controlledUser = random.sample(set(range(data.user_num)), self.fakeUserNum)

        self.attackForm = "gradientAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        self.gradMaxLimitation = int(arg.gradMaxLimitation)
        self.gradNumLimitation = int(arg.gradNumLimitation)
        self.BiLevelOptimizationEpoch = 4
        self.attackEpoch = int(arg.attackEpoch)

    def gradientattack(self, recommender):
        for i in range(self.BiLevelOptimizationEpoch):
            user_embed, item_embed, usergrad, itemgrad = recommender.train(requires_embgrad=True, Epoch=self.attackEpoch)
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