__author__ = 'wangyichao'

from sklearn import svm
import pickle
from sklearn import linear_model

class SvmManager():

    def __init__(self, model_file, verbose=True, isSgd=False, proba=False):
        if isSgd:
            if proba:
                self.clf = linear_model.SGDClassifier(verbose=verbose, loss='log', n_jobs=-1)
            else:
                self.clf = linear_model.SGDClassifier(verbose=verbose, n_jobs=-1)
        else:
            self.clf = svm.SVC(verbose=verbose, probability=proba)
        self.model_file = model_file

    def fit(self, train_list, label_list, store=True):
        self.clf.fit(train_list, label_list)
        if store:
            pickle.dump(self.clf, self.model_file)

    def load(self):
        self.clf = pickle.load(self.model_file)

    def predict(self, query_list):
        return self.clf.predict(query_list)

    def predict_proba(self, query_list, n_predict):
        # print(zip(self.clf.classes_, self.clf.predict_proba(query_list)[0]))
        predict_list = self.clf.predict_proba(query_list)
        return [x.argsort()[:-1][:n_predict] for x in predict_list]

    def decision_function(self, query_list, n_predict):
        predict_list = self.clf.decision_function(query_list)
        return [x.argsort()[::-1][:n_predict] for x in predict_list]

# train_points = [[0, 0, 0], [1, 1,1], [2, 3, 0]] * 10
# train_classes = [0, 1, 2] * 10
# test_point = [[2, 2, 1], [0,0,1]]
# with open("../model_store/svm_model.pkl","wb") as handle:
# svmM = SvmManager(None, isSgd=True)
# svmM.fit(train_points, train_classes, store=False)
# with open("../model_store/svm_model.pkl","rb") as handle:
#     svmM = SvmManager(handle)
#     svmM.load()
# print(svmM.decision_function(test_point, 2))


