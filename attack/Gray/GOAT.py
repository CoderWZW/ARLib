import numpy as np
import random
import torch
import torch.nn as nn
from util.algorithm import l2
from util.tool import targetItemSelect
from scipy.sparse import vstack,csr_matrix


class GOAT():
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
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)
        self.maliciousUserSize = arg.maliciousUserSize
        self.targetSize = arg.targetSize
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.G = None
        self.D_r = None
        self.item_itemInteract = self.interact.T @ self.interact
        self.item_itemInteract[self.item_itemInteract > 0] = 1
        self.itemIntNum = self.item_itemInteract.sum(0).tolist()[0]

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False
        self.BiLevelOptimizationEpoch = 50

    def posionDataAttack(self, epoch1=20, epoch2=20, O_u=0.01, O_g=0.1, O_i=0.02):
        if self.G is None:
            # print("hava no trained Generator, Generator training")
            k = self.maliciousFeedbackNum
            self.k = k
            self.G = Encoder(k).cuda()
            self.D = Decoder(k).cuda()
            optimize_G = torch.optim.Adam(self.G.parameters(), lr=0.005)
            optimize_D = torch.optim.Adam(self.D.parameters(), lr=0.005)
            for i in range(self.BiLevelOptimizationEpoch):
                self.G.eval()
                self.D.train()
                for k1 in range(epoch1):
                    I_s, I_f, realUserList = self.itemSample(k, O_u, O_g, O_i)
                    realUserMat = torch.tensor(realUserList).float()
                    fakeUserMat = torch.zeros_like(realUserMat)
                    Z = torch.randn(fakeUserMat.shape)
                    fakeRatings = self.G(Z.cuda())
                    loss1 = (self.D(fakeRatings) - self.D(realUserMat.cuda())).mean()
                    optimize_D.zero_grad()
                    loss1.backward()
                    optimize_D.step()
                    # print("epoch{} miniepoch{} D:{}".format(i, k1, loss1))
                self.D.eval()
                self.G.train()
                for k2 in range(epoch2):
                    I_s, I_f, realUserList = self.itemSample(k, O_u, O_g, O_i)
                    realUserMat = torch.tensor(realUserList).float()
                    fakeUserMat = torch.zeros_like(realUserMat)
                    Z = torch.randn(fakeUserMat.shape)
                    fakeRatings = self.G(Z.cuda())
                    loss2 = (-self.D(fakeRatings) + 0.01 * (1 / self.k) * torch.linalg.norm(
                        fakeRatings - realUserMat.cuda())).mean()
                    optimize_G.zero_grad()
                    loss2.backward()
                    optimize_G.step()
                    # print("epoch{} miniepoch{} G:{}".format(i, k2, loss2))
        self.G.eval()
        I_s, I_f, realUserList = self.itemSample(self.k, O_u, O_g, O_i)
        realUserMat = torch.tensor(realUserList).float()
        fakeUserMat = torch.zeros_like(realUserMat)
        Z = torch.randn(fakeUserMat.shape)
        fakeRatings = self.G(Z.cuda()).cpu()
        fakeRat = torch.zeros((fakeRatings.shape[0], self.itemNum))
        for step, i in enumerate(realUserList):
            fakeRat[step, I_s[step] + I_f[step]] = fakeRatings[step, :]
            fakeRat[step, self.targetItem] = 1
        fakeRat = self.project(fakeRat, self.maliciousFeedbackNum)
        self.t = fakeRat
        # print("tureScore:{}".format(self.D(fakeRatings.cuda()).mean()))
        return vstack([self.interact, csr_matrix(fakeRat.detach())])

    def project(self, mat, n):
        matrix = torch.tensor(mat)
        _, indices = torch.topk(matrix, n, dim=1)
        matrix.zero_()
        matrix.scatter_(1, indices, 1)
        return matrix
    
    def itemSample(self, k, O_u, O_g, O_i):
        I_s = []
        I_f = []
        realUserList = []
        ratingNumThreshold = int(O_i * self.userNum)
        ps = 0.3
        realUser = np.zeros((1, self.itemNum))
        for i in range(self.fakeUserNum):
            I_s.append([])
            I_f.append([])
            while realUser.sum() < O_u * self.itemNum:
                ind = random.randint(0, self.userNum - 1)
                realUser = self.interact[ind, :].toarray()[0,:]
            if k == 0:
                k = min(realUser.sum(), int(O_g * self.itemNum))
            itemSet = realUser.nonzero()[0].tolist()
            for j in itemSet:
                if self.itemIntNum[j] > ratingNumThreshold and len(I_s[-1]) < int(k * 0.3):
                    I_s[-1].append(j)
                elif self.itemIntNum[j] > ratingNumThreshold / 3 and len(I_f[-1]) < int(k * 0.7):
                    I_f[-1].append(j)
            while len(I_s[-1]) < int(k * 0.3):
                I_s[-1] += random.sample(
                    set(list(range(self.itemNum))) - set(self.targetItem) - set(I_s[-1]) - set(I_f[-1]),
                    int(k * 0.3) - len(I_s[-1]))
            while len(I_f[-1]) + len(I_s[-1]) < k:
                I_f[-1] += random.sample(
                    set(list(range(self.itemNum))) - set(self.targetItem) - set(I_s[-1]) - set(I_f[-1]),
                    k - len(I_f[-1]) - len(I_s[-1]))
            realUserList.append(realUser[I_s[-1] + I_f[-1]])
        return I_s, I_f, realUserList


class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSize, sigmoidFunc=False):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(hiddenSize)):
            if i == 0:
                self.net.add_module('layer_{}'.format(i), nn.Linear(inputSize, hiddenSize[0]))
            else:
                self.net.add_module('layer_{}'.format(i), nn.Linear(hiddenSize[i - 1], hiddenSize[i]))
            if not sigmoidFunc:
                self.net.add_module('bias_{}'.format(i), nn.LeakyReLU(0.2))
            else:
                self.net.add_module('bias_{}'.format(i), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, k):
        self.k = k
        super(Encoder, self).__init__()
        self.G_e = MLP(k, [64, 32, 16 * k])
        self.G_l = MLP(k, [64, 32, 16])
        self.G_r = MLP(k * 16, [k])

    def forward(self, x):
        L_t = self.G_l(x)
        H = self.G_e(x).reshape(x.shape[0], self.k, 16)
        L = L_t.T @ L_t
        R_t1 = torch.bmm(H, L.unsqueeze(0).repeat((x.shape[0], 1, 1)))
        R_t2 = self.G_r(R_t1.view((-1, R_t1.shape[1] * R_t1.shape[2])))
        return R_t2


class Decoder(nn.Module):
    def __init__(self, k):
        super(Decoder, self).__init__()
        self.D_r = MLP(k, [64, 32, 16, 1], sigmoidFunc=True)

    def forward(self, x):
        return self.D_r(x)
