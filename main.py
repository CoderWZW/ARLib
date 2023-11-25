import time
from conf.attack_parser import attack_parse_args
from conf.recommend_parser import recommend_parse_args
from util.DataLoader import DataLoader
from util.tool import seedSet
from ARLib import ARLib
import os
import torch
import numpy as np
import random


if __name__ == '__main__':

    # 1. Load configuration
    recommend_args = recommend_parse_args()
    attack_args = attack_parse_args()
    # 2. Import recommend model and attack model
    os.environ['CUDA_VISIBLE_DEVICES'] = recommend_args.gpu_id
    seed = recommend_args.seed
    seedSet(seed)

    import_str = 'from recommend.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)
    import_str = 'from attack.' + attack_args.attackCategory + "." + attack_args.attackModelName + ' import ' + attack_args.attackModelName
    exec(import_str)

    # 3. Load data
    data = DataLoader(recommend_args)

    # 4. Define recommend model and attack model, and define ARLib to control the process
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    attack_model = eval(attack_args.attackModelName)(attack_args, data)
    arlib = ARLib(recommend_model, attack_model, recommend_args, attack_args)

    s = time.time()

    # 5. Train and test in clean data (before attack)
    arlib.RecommendTrain()
    arlib.RecommendTest()
    # 6. Attack
    # generate poison data, and then train/test in poisoning data (after attack)
    arlib.PoisonDataAttack()
    for step in range(arlib.times):
        print("attack step:{}".format(step))
        # seedSet(seed)
        arlib.RecommendTrain(attack=step)
        arlib.RecommendTest(attack=step)

    # 7. N times experimental results analysis (on average)
    arlib.ResultAnalysis()

    e = time.time()
    print("Running time: %f s" % (e - s))
