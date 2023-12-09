import os
import time
import gym
from gym import spaces
from gym.envs.registration import register
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli
from stable_baselines3 import PPO
import numpy as np
import scipy.sparse as sp
import random
from util.tool import targetItemSelect
from util.metrics import AttackMetric
from scipy.sparse import vstack, csr_matrix
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy, MultiInputActorCriticPolicy, partial
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.distributions import Distribution


class PoisonRec():
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
            # policy_kwargs = dict(features_extractor_class=CustomFeaturesExtractor)
            self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
            self.agent = PPO(CustomPolicy, self.env, verbose=1, clip_range=0.1, gamma=1,n_steps=20,n_epochs=10)
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
        self.observation_space = spaces.Dict({"userId":spaces.Discrete(self.fakeUserNum), "itemInteract":spaces.MultiBinary(item_num)})
        # self.observation_space = spaces.MultiBinary(item_num)
        # self.action_space = spaces.Discrete(item_num) # 例如，离散动作空间
        self.action_space = spaces.MultiBinary(item_num) # 例如，离散动作空间

        # self.state_dim = self.observation_space.shape  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
    
        self.if_discrete = True
        self.itemList = self.targetItem
        # self.state = [0, np.zeros(self.item_num)]
        # self.state[1][self.itemList] = 1
        self.state = {"userId":0, "itemInteract":np.zeros(self.item_num)}
        self.state["itemInteract"][self.itemList] = 1
        self.fakeUserDone = False
        self.fakeUserid = 0

    def reset(self):
        # 重置环境
        self.itemList = self.targetItem
        self.reward = 0
        if self.fakeUserDone:
            self.fakeUserDone = False
            self.fakeUserid = 0
        self.state = {"userId":0, "itemInteract":np.zeros(self.item_num)}
        self.state["itemInteract"][self.itemList] = 1
        return self.state 

    def step(self, action): 
        ones_indices = np.where(action == 1)[0]
        if len(ones_indices) > self.maliciousFeedbackNum:
            keep_indices = np.random.choice(ones_indices, size=self.maliciousFeedbackNum, replace=False)
            action = np.zeros_like(action)
            action[keep_indices] = 1
        self.state["itemInteract"][np.where(action == 1)[0]] = 1
        self.state["itemInteract"][self.targetItem] = 1
        self.fakeUserInjectChange(self.recommender, self.fakeUserid, self.itemList)
        attackmetrics = AttackMetric(self.recommender, self.targetItem, [50])
        reward = attackmetrics.hitRate() * self.recommender.data.user_num
        done = True
        if self.fakeUserid == self.fakeUserNum - 1: self.fakeUserDone = True
        self.fakeUserid = (self.fakeUserid + 1) % self.fakeUserNum
        self.state["userId"] = self.fakeUserid
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

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.user_embedding = nn.Embedding(observation_space.spaces["userId"].n, features_dim)
        self.item_embedding = nn.EmbeddingBag(observation_space.spaces["itemInteract"].n, features_dim)

    def forward(self, observations):
        E_u = self.user_embedding(observations["userId"].long())
        E_i = self.item_embedding(observations["itemInteract"].long())
        if E_u.dim() == 2:
            E_u = E_u.unsqueeze(0)
        E_i = E_i.unsqueeze(1)
        return torch.vstack((E_u, E_i)), self.item_embedding.weight

class CustomPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.observation_space = args[0]
        self.action_space = args[1]
        super(CustomPolicy, self).__init__(*args, **kwargs)
        

    def forward(self,obs ,deterministic=False):
        features = self.extract_features(obs)
        E_st, E_item = self.lstm_extractor(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(E_st, E_item)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.action_dist = BernoulliDistributionEx(self.action_space.n)

        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_lstmnet(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.lstm_extractor = CustomFeaturesExtractor(self.observation_space)
    def _get_action_dist_from_latent(self, E_st, E_item) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(E_st, E_item)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        E_st, E_item = self.lstm_extractor(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(E_st, E_item)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def get_distribution(self, obs):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        E_st, E_item = self.lstm_extractor(obs)
        features = super().extract_features(obs, self.pi_features_extractor)
        return self._get_action_dist_from_latent(E_st, E_item)

class BernoulliDistributionEx(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dims: int):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution_lstmnet(self, latent_dim) -> nn.Module:
        return LSTMNet(latent_dim)

    def proba_distribution(self, action_logits: th.Tensor):
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.round(self.distribution.probs)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor):
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

class LSTMNet(nn.Module):
    def __init__(self,feature_dim = 64):
        super(LSTMNet, self).__init__()
        # Policy network
        self.dim = feature_dim
        self.DNN = nn.Sequential(nn.Linear(self.dim, self.dim),
        nn.ReLU(), nn.Linear(self.dim, self.dim), nn.ReLU())
        self.LSTM = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=2)

    def forward(self, E_st, E_item):
        h_t, _ = self.LSTM(E_st)
        output = torch.softmax(self.DNN(h_t[-1,:,:]) @ E_item.T,dim=1)
        return output
