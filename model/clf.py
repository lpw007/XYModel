# -*-coding:GBK -*-
from sklearn.model_selection import StratifiedKFold
from .metrics import *
# import torch
# import time
# import os
from .models import *
import numpy as np

class BaseClf(object):
    def __init__(self):
        pass

    def calculate_y_logit(self, X, XLen):
        pass

    def cv_train(self,data_class, train_size=256, batch_size=256, epoch=100, stopRounds=10, early_stop=10, save_rounds=1,
                 optim='Adam', lr=0.001, weight_decay=0, folds=5, isHigherBetter=True, metrics="MaF", report=["ACC", "MaF"],
                 save_path='model'):
        skf = StratifiedKFold(n_splits=folds)
        validRes, testRes = [], []
        for i, (train_indices, valid_indices) in enumerate(skf.split(data_class.train_data,data_class.label)):
            print(f'CV_{i+1}:')
            self.reset_parameters()  # 参数初始化 相当于训练新的模型
            data_class.train_id_list = train_indices
            data_class.valid_id_list = valid_indices
            res = self.train(data_class,train_size,batch_size,epoch,stopRounds,early_stop,save_rounds,optim,lr,weight_decay,
                             isHigherBetter,metrics,report,f"{save_path}_cv{i+1}")
            validRes.append(res)
            Metrictor.table_show(validRes, report)

    def train(self,data_class,train_size,batch_size,epoch,stopRounds,early_stop,save_rounds,optim,lr,weight_decay,
              isHigherBetter=True, metrics="MaF", report=["ACC", "MaF"],
              save_path='model'):
        assert batch_size%train_size == 0
        metrictor = Metrictor(data_class.classNum)
        self.stepCounter = 0
        self.stepUpdate = batch_size // train_size
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weight_decay)
        train_stream = data_class.random_batch_data_stream(batch_size=train_size, type='train', device=self.device)
        itersPerEpoch = (len(data_class.train_id_list)+train_size-1)//train_size
        mtc, bestMtc, stopSteps = 0.0, 0.0, 0
        if len(data_class.valid_id_list) > 0:
            valid_stream = data_class.random_batch_data_stream(batch_size=train_size, type='valid', device=self.device)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(train_stream)
                loss = self._train_step(X,Y,optimizer)
                if stopRounds > 0 and (e * itersPerEpoch + i + 1) % stopRounds == 0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if len(data_class.valid_id_list) > 0:
                        X,Y = next(valid_stream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')

            if len(data_class.valid_id_list) > 0 and (e + 1)%save_rounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                Y_pre, Y = self.calculate_y_prob_by_iterator(\
                    data_class.one_epoch_batch_data_stream(train_size, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y)
                print(f'[Total Train]', end='')
                metrictor(report)
                print(f'[Total Valid]', end='')
                Y_pre, Y = self.calculate_y_prob_by_iterator( \
                    data_class.one_epoch_batch_data_stream(train_size, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
                    print(f' Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl" % save_path, e + 1, bestMtc, data_class)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps >= early_stop:
                        print(
                            f'The val {metrics} has not improved for more than {early_stop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl" % save_path)
        os.rename("%s.pkl" % save_path, "%s_%s.pkl" % (save_path, ("%.3lf" % bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre, Y = self.calculate_y_prob_by_iterator(\
            data_class.one_epoch_batch_data_stream(train_size, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]', end='')
        metrictor(report)
        Y_pre, Y = self.calculate_y_prob_by_iterator(\
            data_class.one_epoch_batch_data_stream(train_size, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]', end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(data_class.id2label)
        print(f'================================')
        return res


    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = \
                dataClass.train_id_list,dataClass.valid_id_list,dataClass.test_id_list
            stateDict['label2id'],stateDict['id2label'] = dataClass.label2id,dataClass.id2label
            stateDict['xy2id'],stateDict['id2xy'] = dataClass.xy2id,dataClass.id2xy
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)

    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.train_id_list = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.valid_id_list = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.test_id_list = parameters['testIdList']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)

    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)

    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)

    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)

    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

    def predict(self,test_dataStream):
        y_pred = []
        while True:
            try:
                X = next(test_dataStream)
            except:
                break
            batch_y_pred = self.calculate_y(X).cpu().data.numpy()
            y_pred.append(batch_y_pred)
        y_pred = np.hstack(y_pred).astype('int32')
        return y_pred

    def pred_prob(self,test_dataStream):
        y_pred = []
        while True:
            try:
                X = next(test_dataStream)
            except:
                break
            batch_y_pred = self.calculate_y_prob(X).cpu().data.numpy()
            y_pred.append(batch_y_pred)
        y_pred = np.vstack(y_pred).astype('float32')
        return y_pred

    def embedding_pre(self,data_stream,type='train'):
        embedding = []
        idx = []
        while True:
            try:
                if type == 'train':
                    X,Y = next(data_stream)
                    idx.append(Y)
                else:
                    X = next(data_stream)
            except:
                break
            X, _ = X['seqArr'], X['seqLenArr']
            X = self.textEmbedding(X)
            X = X.transpose(1, 2)
            # X = self.textSPP(X) # => batchSize × feaSize × sppSize
            X = self.textCNN(X)  # => batchSize × scaleNum*filterNum
            X = self.fcLinear(X)
            X = X.cpu().data.numpy()
            embedding.append(X)
        embedding = np.vstack(embedding).astype('float32')
        if type == 'train':
            idx = np.hstack(idx).astype('int32')
            return embedding,idx
        else:
            return embedding


class TextClassifier_SPPCNN(BaseClf):
    def __init__(self,classNum, embedding, SPPSize=128, feaSize=512, filterNum=448, contextSizeList=[1,3,5], hiddenList=[],
                 embDropout=0.3, fcDropout=0.3,
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float), dropout=embDropout).to(device)
        # self.textSPP = TextSPP(SPPSize).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        # self.textAYNICNN = TextAYNICNN(featureSize=feaSize,contextSizeList=contextSizeList,filterSize=filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList) * filterNum, 3, hiddenList, fcDropout).to(device)
        # self.fcLinear1 = MLP(10, classNum, hiddenList, dropout=0.3,name='MLP1').to(device)
        # self.fcLinear = MLP(feaSize+filterNum*len(contextSizeList),classNum,hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textCNN, self.fcLinear])

        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight,gama=4)

    def calculate_y_logit(self, X):
        X,_ = X['seqArr'],X['seqLenArr']
        # X: batchSize × seqLen
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        X = X.transpose(1,2)
        # X = self.textSPP(X) # => batchSize × feaSize × sppSize
        X = self.textCNN(X) # => batchSize × scaleNum*filterNum
        # X = self.fcLinear(X)
        return self.fcLinear(X) # => batchSize × classNum

