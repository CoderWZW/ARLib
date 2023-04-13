import argparse

def attack_parse_args():
    parser = argparse.ArgumentParser(description='Attack model parameter initialization')

    # ===== Genneral parameters ===== #
    parser.add_argument('--attackModelName', type=str, default="CLRecAttack", metavar='N',
                        choices=["RandomRankingAttack","BandwagonRankingAttack","AUSH","GOAT","FedRecAttack","CLRecAttack"],help='the name of attack.')
    parser.add_argument('--maliciousUserSize', type=float, default="0.01", metavar='N',
                        help='proportion/number of users. proportion if value is float (0.01, 0.1), number if value is int (5, 10).')
    parser.add_argument('--maliciousFeedbackSize', type=float, default="0", metavar='N',
                        help='the feedback number of each user. If value is 0, the number of feedback is the average of real users.')
    parser.add_argument('--attackEpoch', type=int, default="10", metavar='N', help='attack epoch')
    parser.add_argument('--times', type=int, default=1, metavar='N', help='the times of attack experiment')
    parser.add_argument('--poisonDatasetOutPath', type=str, default="data/poison/", metavar='N', help='the poisoning data sava path after attack.')
    parser.add_argument('--poisondataSaveFlag', type=bool, default=True, metavar='N', help='whether to save the attack data result.')

    # ===== Shilling attack parameters ===== #
    parser.add_argument('--selectSize', type=float, default="0.01", metavar='N',
                        help='proportion of select item')
    
    # ===== If attack is gradient attack, the following parameters are necessary ===== #
    parser.add_argument('--gradMaxLimitation', type=int, default="1", metavar='N',
                        help='item grad can not be more than grad_max_limitation')
    parser.add_argument('--gradNumLimitation', type=int, default="60", metavar='N',
                        help='the number of item grad change can not be more than grad_num_limitation')
    parser.add_argument('--gradIterationNum', type=int, default="10", metavar='N',
                        help='the number of item grad change can not be more than grad_num_limitation')

    # ===== If recommend model is rating, the following parameters are necessary ===== #
    parser.add_argument('--maxScore', type=float, default=1, metavar='N',
                        help='Maximum score')
    parser.add_argument('--minScore', type=float, default=0, metavar='N',
                        help='Minimum score')

    # ===== If attack type is target, the following parameters are necessary ===== #
    parser.add_argument('--attackTarget', type=str, default="target", metavar='N', choices=["target", "untarget"],
                        help='attackTarget')
    parser.add_argument('--attackType', type=str, default="push", metavar='N', choices=["push", "nuke"],
                        help='attackType push or nuke')
    parser.add_argument('--attackTargetChooseWay', type=str, default="random", metavar='N',
                        choices=["random", "popular", "unpopular"],
                        help='Target attack selection mode')
    parser.add_argument('--targetSize', type=float, default=5, metavar='N',
                        help='proportion of targetItem, ratio if value is float,number if value is int')

    args = parser.parse_args()
    return args
