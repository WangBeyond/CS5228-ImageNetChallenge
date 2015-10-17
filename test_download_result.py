import os
from scipy.io import loadmat

def main():
    # Iterate over image ranges
    import urllib

    input_file = open("../resource/imagenet.sbow.obtain_synset_list partitioned.txt","r")
    result_file = open("../resource/check_result.txt","w")

    counter = -1
    for line in input_file:
        counter += 1
        if not line:
            break
        if counter >= 400:
            break
        wid = line.strip()
        fname = "../target/%s.sbow.mat"%(wid)
        if not os.path.isfile(fname):
            result_file.write(wid+": missing\n")
        else:
            try:
                testObj = loadmat(fname)
                result = testObj['image_sbow']
                result_file.write(wid+" "+str(result.shape)+"\n")
            except Exception, e:
                result_file.write(wid+"error\n")
            else:
                pass
            finally:
                pass
            
            
    input_file.close()
    result_file.close() 

    
        

if __name__ == '__main__':
    main()