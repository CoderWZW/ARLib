import math
import numpy as np

class RecommendMetric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num/total_num

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = RecommendMetric.hits(origin, predicted)
        hr = RecommendMetric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = RecommendMetric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = RecommendMetric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = RecommendMetric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = RecommendMetric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = RecommendMetric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure


class AttackMetric(object):
    """
    param 
    targetItem:list, targetItem: id
    """
    def __init__(self, recommendModel, targetItem, top=[10]):
        self.recommendModel = recommendModel
        self.targetItem = targetItem
        self.top = top

    def precision(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += k
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def hitRate(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += 1
            for k in range(len(self.top)):
                hit[k] += int(len(set(self.targetItem) & set(result[k])) > 0)/len(self.targetItem)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def recall(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += len(self.targetItem)
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def NDCG(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                idcg=0
                for s in range(k):
                    if s < len(self.targetItem):
                        idcg+= 1 / np.log2(2 + s)
                totalNum[n] += idcg
            for step,r in enumerate(result):
                for rank,j in enumerate(r):
                    if j in self.targetItem:
                        hit[step]+=1 / np.log2(2 + rank)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

