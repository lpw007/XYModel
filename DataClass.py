# -*-coding:GBK -*-
from gensim.models import Word2Vec
from tqdm import tqdm
import os,random,torch

import pandas as pd
import glob

import numpy as np

# trn_path = 'hy_round1_train_20200102'
# test_path = 'hy_round1_testB_20200221'

# label_dict = {'����': 0, '����': 1, 'Χ��': 2}

# ���[1-100]
def csv2strList(df,mesh_size=2000): # mesh_size ���� Ĭ�� 100*100����
    """
        ��csv�ļ��Ĺ켣ת��Ϊ�ַ����б�
    :param cvs_file:
    :param params:
    :return:
    """
    all_x_max,all_x_min,all_y_max,all_y_min = 7133785.482740337,5000249.625693835,7667580.57052393,3345433.07253925 # �����Ǳ߽磬�����ֵ���Ĵ�һ�㼴�ɣ���ȡ����0

    x_new = ((df['x']-all_x_min)/500).values
    x_new = np.ceil(x_new).astype(int)
    y_new = ((df['y']-all_y_min)/500).values
    y_new = np.ceil(y_new).astype(int)
    df_new = pd.DataFrame([x_new,y_new]).T
    df_new["loc"] = df_new[0].map(str)+str('*')+ df_new[1].map(str) # ����*�Ŵ��棬������Ҫ�滻
    new_loc =str(list(df_new["loc"].values)).replace(',', ' ').strip('[]').replace('*',',').replace("'",'') # ȥ�� , [] '
    return new_loc.split('  ')


class DataClassNew(object):
    def __init__(self,trn_path,test_path):
        """
            ��ʼ��������  �ڴ�����������ݵ���ɢ����embedding����
        :param trn_path: ѵ������·��
        :param test_path: ��������·��
        :param method:  embedding����
        :param feaSize:  embedding���ά��
        :param window:  ������С
        :param sg:
        :param workers:
        """
        train_data = []
        label = []
        test_data = []
        self.vector = {}
        self.train_id_list = None
        self.valid_id_list = None
        self.test_id_list = None
        self.train_name_list = list()

        # �ڴ˼�������ѵ������
        for file in glob.glob(trn_path + '/*'):
            df = pd.read_csv(file)
            df = df.iloc[::-1]
            label.append(df['type'][0])
            str_list = csv2strList(df,mesh_size=10000)
            train_data.append(str_list)

            filename = os.path.basename(file)
            idx = filename.split('.')[0]
            self.train_name_list.append(idx)
        self.train_name_list = np.array(self.train_name_list,dtype='int32')



        # ����Ϊ����˳��������ݼ�
        # file_count = 9000
        # test_file_list = os.listdir(test_path)
        # for i in range(len(test_file_list)):
        #     file_base = file_count + i
        #     file_name = test_path + '/' + str(file_base) + '.csv'
        #     df = pd.read_csv(file_name)
        #     df = df.iloc[::-1]
        #     str_list = csv2strList(df, mesh_size=10000)
        #     test_data.append(str_list)

        # �ڴ˼������в�������
        for file in glob.glob(test_path + '/*'):
            df = pd.read_csv(file)
            df = df.iloc[::-1]
            str_list = csv2strList(df,mesh_size=10000)
            test_data.append(str_list)

        data = train_data + test_data

        print('Getting the mapping variables for label and label id......')
        self.label2id, self.id2label = {}, []
        cnt = 0
        for lab in tqdm(label):
            if lab not in self.label2id:
                self.label2id[lab] = cnt
                self.id2label.append(lab)
                cnt += 1
        self.classNum = cnt
        self.label = np.array([self.label2id[i] for i in label], dtype='int32')  # �����б�ǩ��ֵ��

        print('Getting the mapping variables for x,y location(for total data)......')
        self.xy2id, self.id2xy = {"<EOS>": 0}, ["<EOS>"]
        str_cnt = 1  # ͳ������xy�Ĳ��ظ��ܸ���
        for xy_list in tqdm(data):
            for xy in xy_list:
                if xy not in self.xy2id:
                    self.xy2id[xy] = str_cnt
                    self.id2xy.append(xy)
                    str_cnt += 1
        self.xyNum = str_cnt

        # tokening for train x,y
        self.train_data = train_data
        self.xySeqLen_train = np.array([len(s) + 1 for s in self.train_data], dtype='int32')  # ���й켣�ĳ����б�
        self.tokenedXY_train = np.array([[self.xy2id[i] for i in s] for s in self.train_data])  # ������xy�ַ�����ֵ��

        # tokening for test x,y
        self.test_data = test_data
        self.xySeqLen_test = np.array([len(s) + 1 for s in self.test_data], dtype='int32')  # ���й켣�ĳ����б�
        self.tokenedXY_test = np.array([[self.xy2id[i] for i in s] for s in self.test_data])  # ������xy�ַ�����ֵ��

        self.vector = {}

    def vectorize(self, method="char2vec", feaSize=64, window=5, sg=1,
                  workers=8):
        """
            �����˶�x y�ַ�������embedding�ķ��� ��������embedding����
        :param method:
        :param feaSize: ��ʾǶ�������ά��
        :param window:
        :param sg:
        :param workers:
        :param loadCache:
        :return:
        """
        if method == 'char2vec':
            data = self.train_data + self.test_data
            doc = [i + ['<EOS>'] for i in data]
            model = Word2Vec(doc, min_count=0, window=window, size=feaSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.xyNum, feaSize), dtype=np.float32)
            for i in range(self.xyNum):
                char2vec[i] = model.wv[self.id2xy[i]]
            self.vector['embedding'] = char2vec
        print('-----embedding���-----------')

    def random_batch_data_stream(self, batch_size=128, type='train', device=torch.device('cpu')):
        """

        :param batch_size:
        :param type:
        :param device:
        :return:
        """
        idList = self.train_id_list if type == 'train' else self.valid_id_list
        X,XLen = self.tokenedXY_train,self.xySeqLen_train
        while True:
            random.shuffle(idList)
            for i in range((len(idList) + batch_size - 1) // batch_size):
                batch_samples = idList[i*batch_size:(i+1)*batch_size]
                xySeqMaxLen = XLen[batch_samples].max()
                yield {
                          "seqArr": torch.tensor([i + [0] * (xySeqMaxLen - len(i)) for i in X[batch_samples]],dtype=torch.long).to(device), \
                          "seqLenArr": torch.tensor(XLen[batch_samples], dtype=torch.int).to(device)
                      }, torch.tensor(self.label[batch_samples], dtype=torch.long).to(device)


    def one_epoch_batch_data_stream(self,batchSize=128, type='valid', device=torch.device('cpu')):
        if type == 'train':
            idList = self.train_id_list
        elif type == 'valid':
            idList = self.valid_id_list
        X, XLen = self.tokenedXY_train, self.xySeqLen_train
        for i in range((len(idList) + batchSize - 1) // batchSize):
            batch_samples = idList[i*batchSize:(i+1)*batchSize]
            xySeqMaxLen = XLen[batch_samples].max()
            yield {
                      "seqArr": torch.tensor([i + [0] * (xySeqMaxLen - len(i)) for i in X[batch_samples]],
                                             dtype=torch.long).to(device), \
                      "seqLenArr": torch.tensor(XLen[batch_samples], dtype=torch.int).to(device)
                  }, torch.tensor(self.label[batch_samples], dtype=torch.long).to(device)


    def one_epoch_batch_train_data_stream(self, batchSize=128, device=torch.device('cpu')):
        X, XLen = self.tokenedXY_train, self.xySeqLen_train
        idList = list(range(len(X)))
        for i in range((len(idList) + batchSize - 1) // batchSize):
            batch_samples = idList[i * batchSize:(i + 1) * batchSize]
            xySeqMaxLen = XLen[batch_samples].max()
            yield {
                      "seqArr": torch.tensor([i + [0] * (xySeqMaxLen - len(i)) for i in X[batch_samples]],
                                             dtype=torch.long).to(device), \
                      "seqLenArr": torch.tensor(XLen[batch_samples], dtype=torch.int).to(device)
                  },self.train_name_list[batch_samples]


    def one_epoch_batch_test_data_stream(self, batchSize=128, device=torch.device('cpu')):
        X,XLen = self.tokenedXY_test,self.xySeqLen_test
        idList = list(range(len(X)))
        for i in range((len(idList) + batchSize - 1) // batchSize):
            batch_samples = idList[i * batchSize:(i + 1) * batchSize]
            xySeqMaxLen = XLen[batch_samples].max()
            yield {
                      "seqArr": torch.tensor([i + [0] * (xySeqMaxLen - len(i)) for i in X[batch_samples]],
                                             dtype=torch.long).to(device), \
                      "seqLenArr": torch.tensor(XLen[batch_samples], dtype=torch.int).to(device)
                  }


# df = pd.read_csv(trn_path+'/0.csv')
# print(csv2strList(df,mesh_size=10000))




