import os
from os.path import join
from scipy.io import loadmat, savemat

def main():
    data_path = "../resource"
    filename = "combined_1_int16.sbow.mat"
    # filename = "combined.sbow.mat"
    # filename = "partition_combined_1_int16.sbow.mat_5000.sbow.mat"
    obj = loadmat(join(data_path, filename))
    outfile = "../output/input_data{0}.txt".format(filename)
    with open(outfile, "w") as f:
        f.write('train_points shape: {0}\n'.format(obj['train_points'].shape))
        f.write('train_classes shape: {0}\n'.format(obj['train_classes'].shape))
        f.write('test_points shape: {0}\n'.format(obj['test_points'].shape))
        f.write('test_classes shape: {0}\n'.format(obj['test_classes'].shape))
        f.write('test_names shape: {0}\n'.format(obj['test_names'].shape))
        f.write('test_name 0: {0}\n'.format(obj['test_names'][0]))
        f.write('train_classes\n')
        test_classes = obj['test_classes'][0]
        for i in range(0, len(test_classes), 10):
            f.write(str(test_classes[i:i+10])[1:-1]+"\n")


if __name__ == '__main__':
    main()