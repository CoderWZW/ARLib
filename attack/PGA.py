import numpy as np
import random
import torch
from copy import deepcopy
from util.tool import targetItemSelect


class PGA():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]
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

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True
        self.BiLevelOptimizationEpoch = 4
        self.attackEpoch = int(arg.attackEpoch)

    def posionDataAttack(self, recommender, epoch=4):
        recommender = deepcopy(recommender)
        for e in range(self.BiLevelOptimizationEpoch):
            lr = epoch / np.sqrt(e + 1)
            grad = recommender.train(requires_adjgrad=True, Epoch=self.attackEpoch)
            s = torch.tensor(self.interact[self.controlledUser, :]).cuda()
            s = s - lr * grad[self.controlledUser, :]
            s = self.projection(s, (s.shape[0] * s.shape[1]) * 0.05)
            ui_adj = np.zeros((self.userNum + self.itemNum, self.userNum + self.itemNum))
            ui_adj[:self.userNum, self.userNum:] = np.array(s.cpu())
            recommender.model._init_uiAdj(ui_adj + ui_adj.T)
        return self.interact

    def projection(self, s, disVal):
        s[s > 1] = 1
        s[s < 0] = 0
        if torch.sum(s) <= disVal:
            return s
        else:
            lower = torch.min(s - 1)
            upper = torch.max(s)
            mu = (lower + upper) / 2
            s2 = s - mu
            s2[s2 > 1] = 1
            s2[s2 < 0] = 0
            while (torch.sum(s2) - disVal) > 0.1:
                if torch.sum(s2) < disVal:
                    upper = mu
                    mu = (mu + lower) / 2
                else:
                    lower = mu
                    mu = (mu + upper) / 2
                s2 = s - mu
                s2[s2 > 1] = 1
                s2[s2 < 0] = 0
            s = s2
            return s
