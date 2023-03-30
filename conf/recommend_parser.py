import argparse


def recommend_parse_args():
    parser = argparse.ArgumentParser()

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="ml-100k", help="Choose a dataset:[FilmTrust, ]")
    parser.add_argument("--data_path", nargs="?", default="data/clean/", help="data path.")
    parser.add_argument("--training_data", nargs="?", default="/train.txt", help="training data path.")
    parser.add_argument("--val_data", nargs="?", default="/val.txt", help="validation data path.")
    parser.add_argument("--test_data", nargs="?", default="/test.txt", help="test data path.")

    # ===== model ===== #
    parser.add_argument('--model_name', type=str, default='GMF', help='[GMF,WRMF,LightGCN,NCF,NGCF,SGL,SimGCL]')
    parser.add_argument('--maxEpoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--emb_size', type=int, default=32, help='embedding size')
    parser.add_argument('--n_layers', type=int, default=2, help='number of gnn layers')
    parser.add_argument('--reg', type=float, default=1e-3, help='regularization weight')
    parser.add_argument('--lRate', type=float, default=0.005, help='learning rate')
    parser.add_argument("--dropout", type=bool, default=True, help="consider  dropout or not")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="ratio of  dropout")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--topK', nargs='?', default='10, 20', help='topK')


    # ===== save model ===== #
    parser.add_argument("--load", type=bool, default=True, help="load existed model or not")
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--save_dir", type=str, default="./modelsaved/", help="output directory for model")

    return parser.parse_args()
