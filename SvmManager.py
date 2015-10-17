__author__ = 'wangyichao'

from sklearn import svm
import pickle


class SvmManager():

    def __init__(self, model_file, verbose=True):
        self.clf = svm.SVC(verbose=verbose)
        self.model_file = model_file

    def fit(self, train_list, label_list, store=True):
        self.clf.fit(train_list, label_list)
        if store:
            pickle.dump(self.clf, self.model_file)

    def load(self):
        self.clf = pickle.load(self.model_file)

    def predict(self, query_list):
        return self.clf.predict(query_list)

# train_points = [[0, 0, 0], [1, 1,1], [2, 3, 0]]
# train_classes = [0, 1, 2]
# test_point = [[2, 2, 1]]
# with open("../model_store/svm_model.pkl","wb") as handle:
#     svmM = SvmManager(handle)
#     svmM.fit(train_points, train_classes)
# with open("../model_store/svm_model.pkl","rb") as handle:
#     svmM = SvmManager(handle)
#     svmM.load()
#     print(svmM.predict(test_point))


