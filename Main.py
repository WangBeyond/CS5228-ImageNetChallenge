__author__ = 'wangyichao'


from os.path import join
from scipy.io import loadmat, savemat
import numpy as np
from SvmManager import SvmManager
from DbManager import DbManager
import time
import logging
import operator

DATA_PATH = "../resource"
verbose = True

def svm_fit_to_file(proba=False):
    start = time.time()
    with open(model_file,"wb") as handle:
        svmManager = SvmManager(handle, verbose=verbose, isSgd=True, proba=proba)
        svmManager.fit(train_points, train_classes)
    end = time.time()
    logging.info("training fit to file finished in {0}s with {1} ".format(end - start, model_file))

def svm_predict_from_file(proba=False, n_predict=5, isDecision=False):
    start = time.time()
    with open(model_file,"rb") as handle:
        svmManager = SvmManager(handle, verbose=verbose, isSgd=True, proba=proba)
        svmManager.load()
        if not proba:
            result = svmManager.predict(test_points)
        elif not isDecision:
            result = svmManager.predict_proba(test_points, n_predict)
        else:
            result = svmManager.decision_function(test_points, n_predict)

    end = time.time()
    logging.info("testing predict from file finished in {0}s with {1}".format(end - start, model_file))
    print("[RESULT LENGTH]", len(result))
    return result


def svm_local(isDecision=False, n_predict=0, n_neighbors=5000, isSgd=True):
    start = time.time()
    db_manager = DbManager()
    db_manager.connect('../db/annotated.db')
    print("Start converting train points")
    train_points_tmp = train_points.tolist()
    train_classes_tmp = train_classes.tolist()
    print("Conversion finished in time {0}s".format(time.time()-start))

    predict_classes = []
    for point_id, point_vector in enumerate(test_points):
        if point_id >= 100:
            break
        if point_id % 10 == 0:
            print("Start query test point {0}".format(point_id))

        neighbors = [int(x) for x in db_manager.query_knn('../db/annotated.db', point_id, n_neighbors=n_neighbors)[0].split()]
        neighbor_points = [train_points_tmp[neighbor_id] for neighbor_id in neighbors]
        neighbor_classes = [train_classes_tmp[neighbor_id] for neighbor_id in neighbors]

        svmManager = SvmManager(None, verbose=verbose, isSgd=isSgd)
        svmManager.fit(np.array(neighbor_points), np.array(neighbor_classes), store=False)
        if not isDecision:
            predict_classes.append(svmManager.predict([point_vector])[0]) #only one class result
        else:
            predict_classes.append(svmManager.decision_function([point_vector], n_predict)[0] )

    end = time.time()
    logging.info("testing local svm finished in {0}s with {1}".format(end - start, model_file))
    return predict_classes


def log_result(result, proba=False):
    with open(result_file,'w') as outfile:
        if not proba:
            for i in range(0, len(result), 10):
                outfile.write(str(result[i:i+10])[1:-1] + "\n")
        else:
            for prob_list in result:
                outfile.write(str(prob_list)[1:-1]+"\n")


def calc_error_rate(n=0):
    with open(truth_file, 'r') as tfile, open(result_file, 'r') as pfile:
        truth_str_list = tfile.readlines()
        predict_str_list = pfile.readlines()
        truth_list = ' '.join(truth_str_list).split()
        predict_list = ' '.join(predict_str_list).split()
        count = 0
        num_comparison = len(predict_list) if 0 == n else n
        for i in range(num_comparison):
            count += 1 if truth_list[i] != predict_list[i] else 0
        return count, len(predict_list), str(float(count)/len(predict_list)*100)+"%"

def calc_error_rate_proba(n=0):
    with open(truth_file, 'r') as tfile, open(result_file, 'r') as pfile:
        truth_str_list = tfile.readlines()
        predict_str_list = pfile.readlines()
        truth_list = ' '.join(truth_str_list).split()
        folder_count_dict = {}
        folder_accuracy_dict = {}
        count = 0
        rounds = 0
        num_comparison = len(predict_str_list) if 0 == n else n
        for i in range(num_comparison):
            # test_folder = test_names[i][0].split('\\')[-1].split('_')[0]
            test_folder = test_names[i].split('\\')[-1].split('_')[0]
            # if test_folder not in good_folders:
            #     continue
            rounds += 1
            if test_folder not in folder_count_dict:
                folder_count_dict[test_folder] = 0
                folder_accuracy_dict[test_folder] = [0,0]
            else:
                folder_accuracy_dict[test_folder][1] += 1;
            if truth_list[i] in predict_str_list[i].split():
                folder_count_dict[test_folder] += 1;
                folder_accuracy_dict[test_folder][0] += 1;
                print(truth_list[i], predict_str_list[i], "*");
            else:
                print(truth_list[i], predict_str_list[i])
                count += 1;
        folder_accuracy_dict = {k : v[0]/float(v[1]) for k, v in folder_accuracy_dict.items()}
        sorted_accuracy = sorted(folder_accuracy_dict.items(), key=operator.itemgetter(1))
        sorted_count = sorted(folder_count_dict.items(), key=operator.itemgetter(1))
        print(sorted_accuracy)
        print(sorted_count)
        print([x[0] for x in sorted_accuracy[-40:]])
        logging.info(sorted_accuracy)
        logging.info(sorted_count)
        return count, rounds, str(float(count)/rounds*100)+"%"

        # return count, len(predict_str_list), str(float(count)/len(predict_str_list)*100)+"%"



def log_test_classes():
    with open(truth_file, 'w') as f:
        for i in range(0, len(test_classes), 10):
            f.write(str(test_classes[i:i+10])[1:-1]+"\n")

def benchmark_knn():
    db_manager = DbManager()
    db_manager.connect()
    train_classes_tmp = train_classes.tolist()
    class_count_dict = {}
    for point_id, point_vector in enumerate(test_points):

        # test_folder = test_names[point_id][0].split('_')[0]
        test_class = test_classes[point_id]
        # if test_folder not in good_folders:
        #     continue
        if point_id >= 1000:
            break
        if point_id % 10 == 0:
            print("Start query test point {0}".format(point_id))
        neighbors = [int(x) for x in db_manager.query_knn(point_id, n_neighbors=n_neighbors)[0].split()]
        neighbor_classes = [train_classes_tmp[neighbor_id] for neighbor_id in neighbors]
        correct_classes = neighbor_classes.count(test_class)
        print( "{0} {1}".format( correct_classes, float(correct_classes)/len(neighbor_classes)))
        if test_class not in class_count_dict:
            class_count_dict[test_class] = [0,0];
        else:
            class_count_dict[test_class][0] += correct_classes
            class_count_dict[test_class][1] += len(neighbor_classes)  #200, 1000, 5000
    print(class_count_dict)


# data_file = "combined.sbow.mat"
# data_file = "combined_1_int16.sbow.mat"
data_file = "data.mat"

# model_file = "../model_store/svm_model_{0}.pkl".format(data_file)
# result_file = "../output/svm_predict_{0}.txt".format(data_file)
# truth_file = "../output/svm_testclasses_{0}.txt".format(data_file)

n_neighbors = 200

model_file = "../model_store/sgd/svm_model_{0}_prob.pkl".format(data_file)
# result_file = "../output/svm_local/sgd/svm_predict_{0}_{1}_decision.txt".format(data_file, n_neighbors)
result_file = "../output/svm_local/sgd/svm_predict_{0}_{1}.txt".format(data_file, n_neighbors)
truth_file = "../output/svm_testclasses_{0}.txt".format(data_file)

# good_folders = {u'n02484975', u'n02123394', u'n13111881', u'n03777568', u'n03868242', u'n03452741', u'n03290653', u'n02981792', u'n03478589', u'n04491638', u'n03337140', u'n04252077', u'n03301568', u'n02328150', u'n02128385', u'n02114367', u'n03417042', u'n02509815', u'n03642806', u'n03742115', u'n04147183', u'n03131574', u'n07743544', u'n03201208', u'n02492660', u'n02480495', u'n02138441', u'n03095699', u'n04273569', u'n02132136', u'n04310018', u'n11978233', u'n03018349', u'n02687172', u'n03218198', u'n02918964', u'n02690373', u'n03447447', u'n03388549', u'n03272562'}


logging.basicConfig(filename='../log/main.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

obj = loadmat(join(DATA_PATH, data_file))

train_points = obj['train_points'] if 'train_points' in obj else obj['train_bows']
train_classes = obj['train_classes'][0] if 'train_classes' in obj else obj['train_image_classes'][0]
test_points = obj['test_points'] if 'test_points' in obj else obj['query_bows']
test_classes = obj['test_classes'][0] if 'test_classes' in obj else obj['query_image_classes'][0]
test_names = obj['test_names'] if 'test_names' in obj else obj['query_image_path']

def main():

    # svm_fit_to_file(proba=True)
    # result = svm_predict_from_file(proba=True, n_predict=5, isDecision=True)
    result = svm_local(isDecision=True, n_predict=5, n_neighbors=n_neighbors, isSgd=True)
    log_result(result, proba=True)
    log_test_classes()
    error_rate = calc_error_rate_proba()
    print(error_rate)
    logging.info("error rate: {0} with result file {1}".format(error_rate, result_file))






if __name__ == '__main__':
    main()