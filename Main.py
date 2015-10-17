__author__ = 'wangyichao'

from os.path import join
from scipy.io import loadmat, savemat
from SvmManager import SvmManager
import time
import logging

DATA_PATH = "../resource"
verbose = True

def svm_fit_to_file():
    start = time.time()
    with open(model_file,"wb") as handle:
        svmManager = SvmManager(handle, verbose=verbose)
        svmManager.fit(train_points, train_classes)
    end = time.time()
    logging.info("training finished in {0} ms".format((end - start)*1000))

def svm_predict_from_file():
    start = time.time()
    with open(model_file,"rb") as handle:
        svmManager = SvmManager(handle, verbose=verbose)
        svmManager.load()
        return svmManager.predict(test_points)
    end = time.time()
    logging.info("testing finished in {0} ms".format((end - start)*1000))

def log_result(result_file, result):
    with open(result_file,'w') as outfile:
        for i in range(0, len(result), 10):
            outfile.write(str(result[i:i+10])[1:-1]+"\n")


def calc_error_rate():
    with open(truth_file, 'r') as tfile, open(result_file, 'r') as pfile:
        truth_str_list = tfile.readlines()
        predict_str_list = pfile.readlines()
        truth_list = ' '.join(truth_str_list).split()
        predict_list = ' '.join(predict_str_list).split()
        count = 0
        for i in range(len(truth_list)):
            count += 1 if truth_list[i] != predict_list[i] else 0
        return count, len(truth_list), str(float(count)/len(truth_list)*100)+"%"

def log_test_classes():
    with open(truth_file, 'w') as f:
        test_classes = obj['test_classes'][0]
        for i in range(0, len(test_classes), 10):
            f.write(str(test_classes[i:i+10])[1:-1]+"\n")

data_file = "combined.sbow.mat"
model_file = "../model_store/svm_model_{0}.pkl".format(data_file)
result_file = "../output/svm_predict_{0}.txt".format(data_file)
truth_file = "../output/svm_testclasses_{0}.txt".format(data_file)
logging.basicConfig(filename='../log/main.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

obj = loadmat(join(DATA_PATH, data_file))
train_points = obj['train_points']
train_classes = obj['train_classes'][0]
test_points = obj['test_points']



# svm_fit_to_file()
result = svm_predict_from_file()
log_result(result)
# log_test_classes()
# print(calc_error_rate())