import time
from conf.attack_parser import attack_parse_args
from conf.recommend_parser import recommend_parse_args
from util.DataLoader import DataLoader
from Manager import Manager
from util.tool import isClass
import attack

if __name__ == '__main__':
    # 1. Load configuration 
    recommend_args = recommend_parse_args()
    attack_args = attack_parse_args()

    # 2. Import recommend model and attack model
    import_str = 'from recommend.' + recommend_args.model_name + ' import ' + recommend_args.model_name
    exec(import_str)
    import_str = 'from attack' + ' import ' + attack_args.attackModelName
    exec(import_str)

    # 3. Load data
    data = DataLoader(recommend_args)

    # 4. Define recommend model and attack model, and define Manager to control the process
    recommend_model = eval(recommend_args.model_name)(recommend_args, data)
    attack_model = eval(attack_args.attackModelName)(attack_args, data)
    manager = Manager(recommend_model, attack_model, recommend_args, attack_args)

    s = time.time()

    # 5. Train and test in clean data (before attack)
    manager.RecommendTrain()
    manager.RecommendTest()

    # 6. Attack
    if attack_model.attackForm == "gradientAttack":
        # perturb gradient of user embeddings or item embeddings during model training and test
        for step in range(manager.times):
            print("attack step:{}".format(step))
            manager.PoisonGradientAttack(attack=step)
            manager.RecommendTest(attack=step)
    elif attack_model.attackForm == "dataAttack":
        # generate poison data, and then train/test in poisoning data (after attack)
        manager.PoisonDataAttack()
        for step in range(manager.times):
            print("attack step:{}".format(step))
            manager.RecommendTrain(attack=step)
            manager.RecommendTest(attack=step)

    # 7. N times experimental results analysis (on average)
    manager.ResultAnalysis()

    e = time.time()
    print("Running time: %f s" % (e - s))
