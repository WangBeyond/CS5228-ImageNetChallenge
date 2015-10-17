__author__ = 'wangyichao'

from os.path import join
from scipy.io import loadmat, savemat
from SvmManager import SvmManager
import time
import logging

DATA_PATH = "../resource"
verbose = False

def svm_fit_to_file(train_points, train_classes, model_file):
    start = time.time()
    with open(model_file,"wb") as handle:
        svmManager = SvmManager(handle, verbose)
        svmManager.fit(train_points, train_classes)
    end = time.time()
    logging.info("training finished {0} ms").format((end - start)*1000)

def svm_predict_from_file(test_points, model_file):
    start = time.time()
    with open(model_file,"rb") as handle:
        svmManager = SvmManager(handle, verbose)
        svmManager.load()
        return svmManager.predict(test_points)
    end = time.time()
    logging.info("testing finished {0} ms").format((end - start)*1000)

def log_result(txtfile, result):
    with open(txtfile,'w') as outfile:
        for i in range(0, len(result), 10):
            outfile.write(str(result[i:i+10])[1:-1]+"\n")

data_file = "combined.sbow.mat"
model_file = "../model_store/svm_model_{0}.pkl".format(data_file)
result_file = "../output/svm_predict_{0}.txt".format(data_file)
logging.basicConfig(filename='../log/main.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

obj = loadmat(join(DATA_PATH, data_file))
train_points = obj['train_points']
train_classes = obj['train_classes'][0]
test_points = obj['test_points']
svm_fit_to_file(train_points, train_classes, model_file)
# result = svm_predict_from_file(test_points, model_file)
# log_result(result_file, result)