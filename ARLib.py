import attack
import recommender
from util.DataLoader import DataLoader
from util.tool import isClass, getPopularItemId, dataSave, targetItemSelect
from util.metrics import AttackMetric
import time
import random
import numpy as np
from time import strftime, localtime, time
from os.path import abspath
import sys
import re
import logging
import os
from shutil import copyfile
from copy import deepcopy
import torch


class ARLib():
    def __init__(self, recommendModel, attackModel, recommendArg, attackArg):
        """
        The recommendation model and attack model are combined to manage the attack experiment
        :param recommendModel: recommender
        :param attackModel: attack model
        :param recommendArg: recommender parser
        :param attackArg: attack parser
        """
        # Evaluation metrics
        self.hitRate = []
        self.precision=[]
        self.recall = []
        self.ndcg = []
        self.RecommendTestResult = []

        # Record the process
        self.result = list()

        # Recommend model setup
        self.recommendModel = recommendModel
        self.recommendModelName = recommendArg.model_name
        self.datasetName = recommendArg.dataset
        self.recommendArg = recommendArg
        self.top = list(map(lambda x: int(x), self.recommendArg.topK.split(",")))

        # Attack model setup
        self.attackModel = attackModel
        self.attackModelName = attackArg.attackModelName
        self.maliciousUserSize = attackArg.maliciousUserSize
        self.maliciousFeedbackSize = attackArg.maliciousFeedbackSize
        self.times = attackArg.times
        self.poisonDatasetOutPath = attackArg.poisonDatasetOutPath

        self.poisondataSaveFlag = attackArg.poisondataSaveFlag  # ！！！not used


        # Target attack setup
        self.attackTargetChooseWay = attackArg.attackTargetChooseWay
        self.targetSize = attackArg.targetSize

        # Gradient attack needs adjacency matrix gradient
        self.requires_grad = self.attackModel.recommenderGradientRequired


        # Determine target items
        self.targetItem = attackModel.targetItem

        # Logger file
        self.current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.logger = logging.getLogger(self.recommendModelName + " attack by " + self.attackModelName)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        self.logFilename = self.recommendModelName + "_" + self.attackModelName + "_" + self.datasetName + "_" + self.attackTargetChooseWay + "_" + \
                           str(self.maliciousUserSize) + "_" + self.current_time
        handler = logging.FileHandler('./log/' + self.logFilename + '.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Basic information in logger file
        message = "\n" * 2 + "-" * 10 + "Recommend Model Infomation" + "-" * 10 + "\n"
        for i in recommendArg._get_kwargs():
            message += str(i[0]) + ":" + str(i[1]) + "\n"
        self.logger.info(message)
        print(message)

        message = "\n" * 2 + "-" * 10 + "Attack Model Infomation" + "-" * 10 + "\n"
        for i in attackArg._get_kwargs():
            message += str(i[0]) + ":" + str(i[1]) + "\n"
        self.logger.info(message)
        print(message)

    def RecommendTrain(self, attack=None):
        """
        train recommender, if data is clean, attack is None, if attack need gradient, requires_grad is True
        """
        if attack is None:
            # Recommend in clean data
            if not os.path.exists(self.recommendArg.save_dir):
                os.makedirs(self.recommendArg.save_dir)
            if not os.path.exists(self.recommendArg.save_dir + self.recommendModelName):
                os.makedirs(self.recommendArg.save_dir + self.recommendModelName)
            if self.recommendArg.load and os.path.exists(
                    self.recommendArg.save_dir + self.recommendModelName + "/" + self.recommendModelName + "_" + str(
                        self.recommendArg.emb_size) + "_"
                    + str(self.recommendArg.n_layers) + "_" + self.datasetName):
                print("Model is exist in {}, loading...".format(
                    self.recommendArg.save_dir + self.recommendModelName + "/" + self.recommendModelName + "_" + str(
                        self.recommendArg.emb_size) + "_"
                    + str(self.recommendArg.n_layers) + "_" + self.datasetName))
                self.recommendModel = torch.load(
                    self.recommendArg.save_dir + self.recommendModelName + "/" + self.recommendModelName + "_" + str(
                        self.recommendArg.emb_size) + "_"
                    + str(self.recommendArg.n_layers) + "_" + self.datasetName)
                self.recommendModel.top = self.recommendArg.topK.split(',')
                self.recommendModel.topN = [int(num) for num in self.recommendModel.top]
                self.recommendModel.max_N = max(self.recommendModel.topN)
            else:
                if self.requires_grad:
                    self.grad = self.recommendModel.train(requires_grad=self.requires_grad)
                else:
                    try:
                        self.recommendModel.train(requires_grad=False)
                    except:
                        self.recommendModel.train()
                if self.recommendArg.save:
                    torch.save(self.recommendModel,
                               self.recommendArg.save_dir + self.recommendModelName + "/" + self.recommendModelName + "_" + str(
                                   self.recommendArg.emb_size) + "_"
                               + str(self.recommendArg.n_layers) + "_" + self.datasetName)
        else:
            # Recommend in poisoning data
            poisonArg = self.recommendArg
            poisonArg.dataset = self.poisonDataName + "/" + str(attack)
            poisonArg.data_path = "data/poison/"
            poisonData = DataLoader(poisonArg)
            Pu, Pi = self.recommendModel.model()
            self.recommendModel.__init__(poisonArg, poisonData)
            if self.requires_grad:
                self.grad = self.recommendModel.train(requires_grad=requires_grad)
            else:
                try:
                    self.recommendModel.train(requires_grad=False)
                except:
                    self.recommendModel.train()

            # torch.save(self.recommendModel,
            #         self.recommendArg.save_dir + self.recommendModelName + "/" + self.recommendModelName + "_" + str(
            #             self.recommendArg.emb_size) + "_"
            #         + str(self.recommendArg.n_layers) + "_" + self.datasetName + "_" + 'attack')

    def RecommendTest(self, attack=None):
        """
        test recommender, if data is clean,attack is None
        """
        if attack is None:
            # recommender test result in clean data
            _, self.rawRecommendresult = self.recommendModel.test()
            message = "Recommender model {} is tested in clean data".format(self.recommendModelName)
            message += "\n" * 2 + "-" * 10 + "Test Result (Evaluation Metrics @Top-({})) in Clean Data".format(
                self.recommendArg.topK) + "-" * 10 + "\n"
            for i in self.rawRecommendresult:
                message += i
            self.logger.info(message)
            print(message)
        else:
            # recommender test  result in poison data
            _, self.attackRecommendresult = self.recommendModel.test()
            self.result.append(dict())
            tempName = "Top 10\n"
            for i in range(len(self.rawRecommendresult)):
                if "Top" in self.rawRecommendresult[i]:
                    tempName = self.rawRecommendresult[i]
                    self.result[-1][tempName] = dict()
                else:
                    self.result[-1][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] = (float(
                        re.sub("[^0-9\.]", "", self.attackRecommendresult[i])) - float(
                        re.sub("[^0-9\.]", "", self.rawRecommendresult[i]))) / float(
                        re.sub("[^0-9\.]", "", self.rawRecommendresult[i]))

            self.RecommendTestResult.append(dict())
            tempName = "Top 10\n"
            for i in range(len(self.rawRecommendresult)):
                if "Top" in self.rawRecommendresult[i]:
                    tempName = self.rawRecommendresult[i]
                    self.RecommendTestResult[-1][tempName] = dict()
                else:
                    self.RecommendTestResult[-1][tempName][
                        re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] = float(
                        re.sub("[^0-9\.]", "", self.attackRecommendresult[i]))

            attackmetrics = AttackMetric(self.recommendModel, self.targetItem, self.top)
            self.hitRate.append(attackmetrics.hitRate())
            self.precision.append(attackmetrics.precision())
            self.recall.append(attackmetrics.recall())
            self.ndcg.append(attackmetrics.NDCG())

            result = dict()
            for i, j in enumerate(self.top):
                result["Top " + str(j)] = dict()
                result["Top " + str(j)]["HitRate"] = self.hitRate[-1][i]
                result["Top " + str(j)]["Precision"] = self.precision[-1][i]
                result["Top " + str(j)]["Recall"] = self.recall[-1][i]
                result["Top " + str(j)]["NDCG"] = self.ndcg[-1][i]

            message = "\n" * 2 + "-" * 10 + "Recommender Test Result in Poisoning Environment No.{} (Evaluation Metrics @Top-({}))". \
                format(attack, self.recommendArg.topK) + "-" * 10 + "\n"
            for i in self.attackRecommendresult:
                message += i
            message += "\n" + "-" * 10 + "Target Attack Test Result in Poisoning Environment No.{} (Evaluation Metrics @Top-({}))". \
                format(attack, self.recommendArg.topK) + "-" * 10
            for i in result.keys():
                message += "\n" + str(i) + "\n"
                for j in result[i].keys():
                    message += str(j) + " : " + str(result[i][j]) + "\n"
            self.logger.info(message)
            print(message)

    def PoisonDataAttack(self):
        """
        Generate PoisonData, path is ./data/poison/
        """
        self.logger.info("Poison Data Generating...")
        print("\n")
        print("_"*80)
        print("Poison Data Generating...")
        self.poisonDataName = self.attackModelName + "_" + self.datasetName +  "_" + \
                              self.attackTargetChooseWay + "_" + str(self.targetSize) + "_" + str(
            self.maliciousUserSize) + "_" + self.current_time
        if not os.path.exists('./data/poison/' + self.poisonDataName):
            os.makedirs('./data/poison/' + self.poisonDataName)
        
        for i in range(self.times):
            if not os.path.exists('./data/poison/' + self.poisonDataName + "/" + str(i)):
                os.makedirs('./data/poison/' + self.poisonDataName + "/" + str(i))

            if self.requires_grad:
                poisonRatings = self.attackModel.posionDataAttack(self.grad)
            elif self.attackModel.recommenderModelRequired:
                poisonRatings = self.attackModel.posionDataAttack(deepcopy(self.recommendModel))
            else:
                poisonRatings = self.attackModel.posionDataAttack()

            # save train.txt,test.txt,val.txt in posion data path
            dataSave(poisonRatings, './data/poison/' + self.poisonDataName + "/" + str(i) + "/" + "train.txt",
                     self.recommendModel.data.id2user, self.recommendModel.data.id2item)
            copyfile(self.recommendArg.data_path + self.recommendArg.dataset + self.recommendArg.val_data,
                     './data/poison/' + self.poisonDataName + "/" + str(i) + "/" + "val.txt")
            copyfile(self.recommendArg.data_path + self.recommendArg.dataset + self.recommendArg.test_data,
                     './data/poison/' + self.poisonDataName + "/" + str(i) + "/" + "test.txt")
            
            self.logger.info("Data attack No.{} by {} has done.".format(i + 1, self.attackModelName))
            print("Data attack No.{} by {} has done.".format(i + 1, self.attackModelName))
        
        print("_"*80)
        print("\n")

    def ResultAnalysis(self):
        """
        After the attack, record experimental results
        """
        self.avgHitRateAttack = []
        for i in range(len(self.hitRate[0])):
            self.avgHitRateAttack.append(sum(map(lambda x: x[i], self.hitRate)) / len(self.hitRate))

        self.avgPrecisionAttack = []
        for i in range(len(self.precision[0])):
            self.avgPrecisionAttack.append(sum(map(lambda x: x[i], self.precision)) / len(self.precision))

        self.avgRecallAttack = []
        for i in range(len(self.recall[0])):
            self.avgRecallAttack.append(sum(map(lambda x: x[i], self.recall)) / len(self.recall))

        self.avgNDCGAttack = []
        for i in range(len(self.ndcg[0])):
            self.avgNDCGAttack.append(sum(map(lambda x: x[i], self.ndcg)) / len(self.ndcg))

        tempName = "Top 10\n"
        for i in range(len(self.rawRecommendresult)):
            if "Top" in self.rawRecommendresult[i]:
                tempName = self.rawRecommendresult[i]
                if len(self.result)!=1:self.result[-1][tempName] = dict()
            elif len(self.result)==1:
                self.result[-1][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] =\
                    self.result[0][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]]
            else:
                self.result[-1][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] = sum(
                    [self.result[j][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] for j in
                     range(len(self.result) - 1)]) / (len(self.result) - 1)

        tempName = "Top 10\n"
        for i in range(len(self.rawRecommendresult)):
            if "Top" in self.rawRecommendresult[i]:
                tempName = self.rawRecommendresult[i]
                if len(self.RecommendTestResult)!=1:self.RecommendTestResult[-1][tempName] = dict()
            elif len(self.RecommendTestResult)==1:
                self.RecommendTestResult[-1][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] =\
                    self.RecommendTestResult[0][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]]
            else:
                self.RecommendTestResult[-1][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] = sum(
                    [self.RecommendTestResult[j][tempName][re.sub("[0-9\.]", "", self.rawRecommendresult[i])[:-1]] for j
                     in
                     range(len(self.RecommendTestResult) - 1)]) / (len(self.RecommendTestResult) - 1)

        message = "\n" * 2 + "-" * 10 + "Recommender Test Result in Poisoning Environment on Average (Evaluation Metrics @Top-({}))".format(
            self.recommendArg.topK) + "-" * 10 + "\n"
        for i in self.RecommendTestResult[-1].keys():
            message += str(i)
            for j in self.RecommendTestResult[-1][i].keys():
                message += str(j) + " : " + str(self.RecommendTestResult[-1][i][j]) + "\n"

        message += "\n" + "-" * 10 + " Global Recommender Performance Variation" + "-" * 10 + "\n"
        for i in self.result[-1].keys():
            message += str(i)
            for j in self.result[-1][i].keys():
                message += str(j) + " : " + str(self.result[-1][i][j]) + "\n"

        result = dict()
        for i, j in enumerate(self.top):
            result["Top " + str(j)] = dict()
            result["Top " + str(j)]["HitRate"] = self.avgHitRateAttack[i]
            result["Top " + str(j)]["Precision"] = self.avgPrecisionAttack[i]
            result["Top " + str(j)]["Recall"] = self.avgRecallAttack[i]
            result["Top " + str(j)]["NDCG"] = self.avgNDCGAttack[i]
        message += "\n" + "-" * 10 + "Target Attack Test Result Poisoning Environment on Average (Evaluation Metrics @Top-({}))".format(
            self.recommendArg.topK) + "-" * 10
        for i in result.keys():
            message += "\n" + str(i) + "\n"
            for j in result[i].keys():
                message += str(j) + " : " + str(result[i][j]) + "\n"
        self.logger.info(message)
        print(message)
