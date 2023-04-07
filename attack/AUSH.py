import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect


class AUSH():
    def __init__(self, arg, data):
        """
        generate fake users by GAN
        :param arg: parameter configuration
        :param data: dataLoader
        """
        self.interact = data.matrix()
        self.userNum = data.user_num
        self.itemNum = data.item_num
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.maliciousUserSize = arg.maliciousUserSize
        self.selectSize = arg.selectSize
        self.targetSize = arg.targetSize
        self.attackType = arg.attackType
        self.maxScore = arg.maxScore
        self.minScore = arg.minScore
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.G = None
        self.D = None

        # The probability that non-target items are sampled
        self.itemP = self.interact.sum(0) / self.interact.sum()
        self.itemP[self.targetItem] = 0
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackSize = (self.interact.sum() / self.userNum) / self.itemNum
            self.selectSize = self.maliciousFeedbackSize / 2
        if self.maliciousUserSize<1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False
        self.BiLevelOptimizationEpoch = 100
        self.attackEpoch = int(arg.attackEpoch)

    def posionDataAttack(self, epoch1=50, epoch2=50):
        """
        posion Data Generate
        :param epoch: Total epoch
        :param epoch1: discriminator update epoch num
        :param epoch2: generator update epoch num
        """
        if self.G is None:
            print("hava no trained Generator, Generator training")
            self.selectItem = random.sample(set(list(range(self.itemNum))) - set(self.targetItem),
                                            int(self.selectSize * self.itemNum))

            G = Generator(self.itemNum).cuda()
            D = Discriminator(self.itemNum).cuda()
            optimize_G = torch.optim.Adam(G.parameters(), lr=0.005)
            optimize_D = torch.optim.Adam(D.parameters(), lr=0.005)
            for i in range(self.BiLevelOptimizationEpoch):
                G.eval()
                D.train()
                for k1 in range(epoch1):
                    userSet = random.sample(set(list(range(self.userNum))),
                                            self.fakeUserNum)
                    #sample item rating
                    tempInteract = np.array(
                        [np.random.binomial(1, self.itemP) for i in range(self.fakeUserNum)])
                    tempInteract = torch.tensor(self.interact[userSet] * tempInteract).float()
                    fakeInteract = G(tempInteract.cuda())
                    loss1 = -(torch.log(D(tempInteract.cuda())).mean() + torch.log(1 - D(fakeInteract.cuda()))).mean()
                    optimize_D.zero_grad()
                    loss1.backward()
                    optimize_D.step()
                    print("epoch{} miniepoch{} D:{}".format(i, k1, loss1))
                D.eval()
                G.train()
                for k2 in range(epoch2):
                    userSet = random.sample(set(list(range(self.userNum))),
                                            self.fakeUserNum)
                    # sample item rating
                    tempInteract = np.array(
                        [np.random.binomial(1, self.itemP) for i in range(self.fakeUserNum)])
                    tempInteract = torch.tensor(self.interact[userSet] * tempInteract).float()
                    fakeInteract = G(tempInteract.cuda())
                    maskTarget = torch.zeros((self.itemNum, 1)).cuda()
                    maskTarget[self.targetItem] = 1
                    if self.attackType == "push":
                        Q = torch.ones_like(fakeInteract) * self.maxScore
                    else:
                        Q = torch.zeros_like(fakeInteract)
                    L_recon = (fakeInteract - tempInteract.cuda()) ** 2
                    L_shill = ((Q.cuda() - fakeInteract) @ maskTarget) ** 2
                    L_GD = torch.log(D(tempInteract.cuda())).mean() + torch.log(1 - D(fakeInteract.cuda())).mean()
                    loss2 = L_GD + L_shill.mean() + L_recon.mean()
                    optimize_G.zero_grad()
                    loss2.backward()
                    optimize_G.step()
                    print("epoch{} miniepoch{} G:{}".format(i, k2, loss2))
            self.G = G
            self.D = D
        self.G.eval()
        userSet = random.sample(set(list(range(self.userNum))),
                                self.fakeUserNum)
        tempInteract = np.array(
            [np.random.binomial(1, self.itemP) for i in range(self.fakeUserNum)])
        tempInteract = torch.tensor(self.interact[userSet] * tempInteract).float()
        fakeRat = self.G(tempInteract.cuda()).detach().cpu().numpy()
        if self.maxScore == 1 and self.minScore == 0:
            fakeRat[fakeRat > 0.5] = 1
            fakeRat[fakeRat <= 0.5] = 0
        print("tureScore:{}".format(self.D(tempInteract.cuda()).mean()))
        return np.vstack([self.interact, fakeRat])


class Generator(nn.Module):
    """
     In AUSH, Generator is MLP
    """
    def __init__(self, size, layer=2):
        self.layer = layer
        super(Generator, self).__init__()
        self.net = nn.Sequential()
        for i in range(self.layer):
            self.net.add_module('layer_{}'.format(i), nn.Linear(size, size))
            if i != self.layer - 1:
                self.net.add_module('bias_{}'.format(i), nn.ReLU())
            else:
                self.net.add_module('bias_{}'.format(i), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
     In AUSH, Discriminator is Linear layer
    """
    def __init__(self, size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(size, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
