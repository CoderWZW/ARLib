import numpy as np
import random
import torch
from util.tool import targetItemSelect


class A_ra():
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
        self.BiLevelOptimizationEpoch = 10
        self.attackEpoch = int(arg.attackEpoch)
        self.n = 50
        self.sigma = 1

    def gradientattack(self, recommender):
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        for i in range(self.BiLevelOptimizationEpoch):
            _, item_embed, usergrad, itemgrad = recommender.train(requires_embgrad=True, Epoch=self.attackEpoch,
                                                                  optimizer=optimizer, evalNum=4)
            item_embed.requires_grad = True
            approximateUserEmb = (torch.randn((self.n, item_embed.shape[1])) * self.sigma).cuda()
            loss = 0
            for i in self.targetItem:
                for j in range(self.n):
                    loss += -torch.log(torch.sigmoid(approximateUserEmb[j, :] @ item_embed[i, :].T) + 10e-8)
            loss.backward(retain_graph=True)
            new_item_grad = item_embed.grad
            user_embed_grad = 0
            recommender.model.attack_emb(user_embed_grad, new_item_grad)
