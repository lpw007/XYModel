# -*-coding:GBK -*-
from DataClass import DataClassNew
import argparse
from model.clf import TextClassifier_SPPCNN
from glob import glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
# parser.add_argument('--k', default='3')
parser.add_argument('--d', default='320')
parser.add_argument('--s', default='64')
parser.add_argument('--f', default='448')
parser.add_argument('--metrics', default='MaF')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--savePath', default='out_new/')
args = parser.parse_args()


def nn_train_test():
    """
    完成nn的训练以及预测
    :return: 得到5cv的csv预测文件
    """
    trn_path = 'hy_round1_train_20200102'
    test_path = 'hy_round1_testB_20200221'
    d, s, f = int(args.d), int(args.s), int(args.f)
    device, path = args.device, args.savePath
    metrics = args.metrics

    report = ["MiF", "MaF"]

    dataClass = DataClassNew(trn_path,test_path)
    dataClass.vectorize("char2vec", feaSize=d)
    # print('-----将embedding保存到npy文件中----------')
    np.save('embedding_320.npy',dataClass.vector['embedding'])
    # dataClass.vector['embedding'] = np.load('embedding_320.npy')

    model = TextClassifier_SPPCNN(classNum=3, embedding=dataClass.vector['embedding'], SPPSize=s, feaSize=d,
                                  filterNum=f,
                                  contextSizeList=[1, 3, 5], hiddenList=[],
                                  embDropout=0.3, fcDropout=0.5, useFocalLoss=True, weight=None,
                                  device=device)

    model.cv_train(dataClass, train_size=16, batch_size=32, stopRounds=-1, early_stop=15,
                   epoch=100, lr=0.001, folds=5, save_path=f'{path}CNN_s{s}_f{f}_d{d}', report=report)

    print('---------------NN训练完成--------------------')
    label_dict = {}
    for i in range(len(dataClass.id2label)):
        print(dataClass.id2label[i])
        label_dict[i] = dataClass.id2label[i]
    print(label_dict)

    num_no = list(range(9000, 11000))
    num_no = np.array(num_no, dtype=int)
    count = 1
    for i in glob('out_new/*'):
        model.load(i)
        model.to_eval_mode()
        y_pre = model.predict(dataClass.one_epoch_batch_test_data_stream(batchSize=64, device=model.device))
        # np.save('prob'+str(count)+'.npy',y_pre)
        # print(y_pre)
        final = pd.DataFrame(y_pre, columns=['label'], index=num_no)
        final['label'] = final['label'].map(label_dict)
        final.to_csv('model' + str(count) + '.csv', header=None, encoding="UTF-8")
        print('写入第' + str(count) + '个')
        count += 1
    print('----------------NN模型预测结束----------------')


