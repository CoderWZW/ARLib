import os
import time
import gym
from gym import spaces
from gym.envs.registration import register
import torch
import torch.nn as nn
from torch import Tensor
from stable_baselines3 import PPO
import numpy as np
import scipy.sparse as sp
import random
from util.tool import targetItemSelect
from util.metrics import AttackMetric
from scipy.sparse import vstack, csr_matrix
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RLAttack():
    def __init__(self, arg, data):
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

        self.env = None
        self.item_num = data.item_num
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

    def posionDataAttack(self,recommender):
        self.recommender = recommender
        self.fakeUserInject(self.recommender)
        if self.env is None:
            self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
            self.agent = PPO('MlpPolicy', self.env, verbose=1, clip_range=0.1, gamma=1,n_steps=20,n_epochs=10)
            self.agent.learn(total_timesteps=400)
        self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
        while not self.env.fakeUserDone:
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _states = self.agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            uiAdj = self.recommender.data.matrix()
            uiAdj[self.fakeUser[ self.env.fakeUserid],:] = 0  
            uiAdj[self.fakeUser[ self.env.fakeUserid],self.env.itemList] = 1
        self.interact = uiAdj
        return self.interact


    def fakeUserInject(self, recommender):
        Pu, Pi = recommender.model()
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)

        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem =  self.targetItem
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
                recommender.user_emb = recommender.model.embedding_dict['user_emb']
                recommender.item_emb = recommender.model.embedding_dict['item_emb'] 
            except:
                recommender.model.embedding_dict['user_mf_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], :Pu.shape[1]//2]
                recommender.model.embedding_dict['user_mlp_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], Pu.shape[1]//2:]
                recommender.model.embedding_dict['item_mf_emb'][:] = Pi[:, :Pi.shape[1]//2]
                recommender.model.embedding_dict['item_mlp_emb'][:] = Pi[:, Pi.shape[1]//2:]

        recommender.model = recommender.model.cuda()



class MyEnv(gym.Env):
    def __init__(self, item_num, fakeUser, maliciousFeedbackNum, recommender, targetItem):
        super(MyEnv, self).__init__()

        self.item_num = item_num
        self.fakeUserNum = len(fakeUser)
        self.recommender = recommender
        self.targetItem = targetItem
        self.maliciousFeedbackNum = maliciousFeedbackNum
        self.fakeUser = fakeUser

        # 定义状态空间和动作空间
        # self.observation_space = spaces.Tuple([spaces.MultiBinary(item_num),spaces.Discrete(self.fakeUserNum)])
        self.observation_space = spaces.MultiBinary(item_num)
        # self.action_space = spaces.Discrete(item_num) # 例如，离散动作空间
        self.action_space = spaces.MultiBinary(item_num) # 例如，离散动作空间

        # self.state_dim = self.observation_space.shape  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
    
        self.if_discrete = True
        self.itemList = self.targetItem
        # self.state = (np.array(self.itemList),0)
        self.state = np.zeros(self.item_num)
        self.state[self.itemList] = 1
        self.fakeUserDone = False
        self.fakeUserid = 0

    def reset(self):
        # 重置环境
        self.itemList = self.targetItem
        self.reward = 0
        if self.fakeUserDone:
            self.fakeUserDone = False
            self.fakeUserid = 0
        # self.state = (np.array(self.itemList), self.fakeUserid)
        # self.state = np.array(self.itemList)
        self.state = np.zeros(self.item_num)
        self.state[self.itemList] = 1
        return self.state 

    def step(self, action): 
        ones_indices = np.where(action == 1)[0]
        if len(ones_indices) > self.maliciousFeedbackNum:
            # 如果值为 1 的元素个数超过了限制，随机选择保留的元素
            keep_indices = np.random.choice(ones_indices, size=self.maliciousFeedbackNum, replace=False)
            action = np.zeros_like(action)
            action[keep_indices] = 1
        self.state[np.where(action == 1)[0]] = 1
        self.state[self.targetItem] = 1
        self.fakeUserInjectChange(self.recommender, self.fakeUserid, self.itemList)
        attackmetrics = AttackMetric(self.recommender, self.targetItem, [50])
        reward = attackmetrics.hitRate()*self.recommender.data.user_num
        done = True
        if self.fakeUserid == self.fakeUserNum - 1: self.fakeUserDone = True
        self.fakeUserid = (self.fakeUserid + 1) % self.fakeUserNum
        # if action not in self.itemList:
        #     self.itemList.append(action)
        #     self.state[action] = 1
        #     # self.state = np.array(self.itemList)
        #     # self.state = (np.array(self.itemList), self.fakeUserid)
        #     if len(self.itemList) >= self.maliciousFeedbackNum:
        #         self.fakeUserInjectChange(self.recommender, self.fakeUserid, self.itemList)
        #         attackmetrics = AttackMetric(self.recommender, self.targetItem, [50])
        #         reward = attackmetrics.hitRate()*self.recommender.data.user_num
        #         done = True
        #         if self.fakeUserid == self.fakeUserNum - 1: self.fakeUserDone = True
        #         self.fakeUserid = (self.fakeUserid + 1)%self.fakeUserNum
        #     else:
        #         reward = 0
        #         done = False
        # else:
        #     reward = -10
        #     done = False
        #     self.state = np.zeros(self.item_num)
        #     self.state[self.itemList] = 1
        info = {}
        return self.state, self.reward, done, info

    def fakeUserInjectChange(self, recommender, fakeUserId, itemList):
        self.userNum = recommender.data.user_num
        self.itemNum = recommender.data.item_num
        uiAdj = recommender.data.matrix()
        uiAdj2 = uiAdj[:, :]
        uiAdj2[self.fakeUser[fakeUserId],:] = 0  
        uiAdj2[self.fakeUser[fakeUserId],self.itemList] = 1
        ui_adj = sp.csr_matrix(([], ([], [])), shape=(
            self.userNum + self.itemNum, self.userNum  + self.itemNum),
                                dtype=np.float32)
        ui_adj[:self.userNum , self.userNum:] = uiAdj2
        recommender.model._init_uiAdj(ui_adj + ui_adj.T)
