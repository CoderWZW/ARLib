import os
import random
import numpy as np
import torch

def isClass(obj, classList):
    """
    If obj is in classList, return True
    else return False
    """
    for i in classList:
        if isinstance(obj, i):
            return True
    return False

def getPopularItemId(interact, n):
    """
    Get the id of the top n popular items based on the number of ratings
    :return: id list
    """
    return np.argsort(interact[:, :].sum(0))[-n:]

def dataSave(ratings, fileName, id2user, id2item):
    """
    sava ratings data
    :param ratings: np.array ratings matrix
    :param fileName: str fileName
    :param id2user: dict
    :param id2item: dcit
    """
    ratingList = []
    ind = ratings.nonzero()
    for i,j in zip(ind[0].tolist(),ind[1].tolist()):
            ratingList.append((i,j,ratings[i,j]))
    # for i in range(ratings.shape[0]):
    #     for j in range(ratings.shape[1]):
    #         if ratings[i,j] == 0: continue
    #         ratingList.append((i, j, ratings[i,j]))
    text = []
    for i in ratingList:
        if i[0] in id2user.keys():
            userId = id2user[i[0]]
        else:
            userId = "fakeUser" + str(i[0])
        itemId = id2item[i[1]]
        new_line = '{} {} {}'.format(userId, itemId, i[2]) + '\n'
        text.append(new_line)
    with open(fileName, 'w') as f:
        f.writelines(text)


def targetItemSelect(data, arg, popularThreshold=0.1):
    interact = data.matrix()
    userNum = interact.shape[0]
    itemNum = interact.shape[1]
    targetSize = arg.targetSize
    if targetSize < 1:
        targetNum = int(targetSize * itemNum)
    else:
        targetNum = int(targetSize)
    path = './data/clean/' + data.dataName + "/" + "targetItem_" + arg.attackTargetChooseWay + "_" + str(
        targetNum) + ".txt"
    if os.path.exists(path):
        with open(path, 'r') as f:
            line = f.read()
            targetItem = [i.replace("'", "") for i in line.split(",")]
        return targetItem
    else:
        def getPopularItemId(n):
            """
            Get the id of the top n popular items based on the number of ratings
            :return: id list
            """
            return np.argsort(interact[:, :].sum(0))[0, -n:].tolist()[0]

        def getReversePopularItemId(n):
            """
            Get the ids of the top n unpopular items based on the number of ratings
            :return: id list
            """
            return np.argsort(interact[:, :].sum(0))[0, :n].tolist()[0]

        if arg.attackTargetChooseWay == "random":
            targetItem = random.sample(set(list(range(itemNum))),
                                       targetNum)
        elif arg.attackTargetChooseWay == "popular":
            targetItem = random.sample(set(getPopularItemId(int(popularThreshold * itemNum))),
                                       targetNum)
        elif arg.attackTargetChooseWay == "unpopular":
            targetItem = random.sample(
                set(getReversePopularItemId(int(0.2 * itemNum))),
                targetNum)
            # targetItem = random.sample(
            #     set(getReversePopularItemId(int((1 - popularThreshold) * itemNum))),
            #     targetNum)
        targetItem = [data.id2item[i] for i in targetItem]
        with open(path, 'w') as f:
            f.writelines(str(targetItem).replace('[', '').replace(']', ''))
        return targetItem

def seedSet(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False