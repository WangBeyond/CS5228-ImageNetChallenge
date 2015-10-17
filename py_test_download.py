__author__ = 'wangyichao'

import os
from urllib2 import urlopen, URLError, HTTPError
import urllib
import thread

NUM_THREAD = 8

# def dlfile(url):
#     # Open the url
#     try:
#         f = urlopen(url)
#         print "downloading " + url
#
#         # Open our local file for writing
#         with open(os.path.basename(url), "wb") as local_file:
#             local_file.write(f.read())
#
#     #handle errors
#     except HTTPError, e:
#         print "HTTP Error:", e.code, url
#     except URLError, e:
#         print "URL Error:", e.reason, url


def read_file(thread_name, delta):
    input_file = open("../resource/imagenet.sbow.obtain_synset_list partitioned.txt","r")
    for line_num, line in enumerate(input_file):
        if not line:
            break
        if line_num % 8 != delta:
            continue
        wid = line.strip()
        fname = "../target/%s.sbow.mat"%wid
        if os.path.isfile(fname) :
            continue
        url = "http://www.image-net.org/downloads/features/sbow/%s.sbow.mat"%wid
        testfile = urllib.URLopener()
        testfile.retrieve(url, fname)
        print(thread_name, "write", line_num)
    input_file.close()
    print(thread_name, "finished")



def main():
    # Iterate over image ranges
    try:
        for delta in range(NUM_THREAD):
            thread.start_new_thread( read_file, ("Thread-" + str(delta), delta) )
    except Exception as e:
       print ("Error: unable to start thread", e)
    while 1:
        pass


if __name__ == '__main__':
    main()