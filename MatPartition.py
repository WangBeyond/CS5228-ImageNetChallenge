__author__ = 'wangyichao'

from os.path import join
from scipy.io import loadmat, savemat

def main():
    data_path = "../resource"
    filename = "combined_1_int16.sbow.mat"
    obj = loadmat(join(data_path, filename))
    train_num = obj['train_points'].shape[0]
    test_num = obj['test_points'].shape[0]
    outfile = "partition_{0}_5000.sbow.mat".format(filename)
    savemat( join(data_path, outfile),
    { "train_points": obj['train_points'][:train_num/10],
    "train_classes":obj['train_classes'][0][:train_num/10],
    "test_points": obj['test_points'][:test_num/10],
    "test_classes": obj['test_classes'][0][:test_num/10],
    })

if __name__ == '__main__':
    main()