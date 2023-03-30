import numpy as np
import random
from util.tool import targetItemSelect


class ShillingAttackModel():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.interact = data.matrix()
        self.userNum = data.user_num
        self.itemNum = data.item_num
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.maliciousUserSize = arg.maliciousUserSize
        self.selectSize = arg.selectSize

        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackSize = (self.interact.sum() / self.userNum) / self.itemNum
            self.selectSize = self.maliciousFeedbackSize / 2
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False

    def getPopularItemId(self, N):
        """
        get N popular items based on the number of feedbacks
        :return: id list
        """
        return np.argsort(self.interact[:, :].sum(0))[-N:]

    def getReversePopularItemId(self, N):
        """
        get N unpopular items based on the number of feedbacks
        :return: id list
        """
        return np.argsort(self.interact[:, :].sum(0))[:N]

    def posionDataAttack(self):
        """
        A rating matrix for the fake user segment is generated according to a specific algorithm
        :return: Fake user rating matrix
        """


class RandomRankingAttack(ShillingAttackModel):
    def __init__(self, arg, data):
        super(RandomRankingAttack, self).__init__(arg, data)

    def posionDataAttack(self):
        uNum = self.fakeUserNum
        fakeRat = np.zeros((uNum, self.itemNum))
        for i in range(uNum):
            fillerItemid = random.sample(set(range(self.itemNum)) - set(self.targetItem),
                                         int(self.maliciousFeedbackSize * self.itemNum) - len(self.targetItem))
            fakeRat[i][fillerItemid] = 1
            # target item in fake user is interact
            fakeRat[i, self.targetItem] = 1
        return np.vstack([self.interact, fakeRat])


class BandwagonRankingAttack(ShillingAttackModel):
    def __init__(self, arg, data):
        super(BandwagonRankingAttack, self).__init__(arg, data)

    def posionDataAttack(self):
        uNum = self.fakeUserNum
        fakeRat = np.zeros((uNum, self.itemNum))
        selectItem = self.getPopularItemId(int(self.selectSize * self.itemNum))
        for i in range(uNum):
            fillerItemid = random.sample(set(range(self.itemNum)) - set(self.targetItem) - set(selectItem),
                                         int(self.maliciousFeedbackSize * self.itemNum) - len(self.targetItem) - len(
                                             selectItem))
            fakeRat[i][fillerItemid] = 1
            fakeRat[i, selectItem] = 1
            fakeRat[i, self.targetItem] = 1
        return np.vstack([self.interact, fakeRat])


class AOPRankingAttack(ShillingAttackModel):
    def __init__(self, arg, data, popularThreshold=0.2):
        super(AOPRankingAttack, self).__init__(arg, data)
        self.popularThreshold = popularThreshold

    def posionDataAttack(self):
        uNum = self.fakeUserNum
        fakeRat = np.zeros((uNum, self.itemNum))
        popularItemId = self.getPopularItemId(int(self.itemNum * self.popularThreshold))
        for i in range(uNum):
            fillerItemid = random.sample(set(popularItemId) - set(self.targetItem),
                                         int(self.maliciousFeedbackSize * self.itemNum) - len(self.targetItem))
            fakeRat[i, fillerItemid] = 1
            fakeRat[i, self.targetItem] = 1
        return np.vstack([self.interact, fakeRat])
