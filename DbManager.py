__author__ = 'wangyichao'

import sqlite3
import logging
import time
from scipy.io import loadmat
import os

class DbManager:


    def connect(self, dbfile):
        self.conn = sqlite3.connect(dbfile)

    def disconnect(self):
        self.conn.close()

    def fill_db_knn(self, dbfile, infile, n_neighbors=5000):


        conn = sqlite3.connect(dbfile)
        c = conn.cursor()
        table_name = "knn_{0}".format(n_neighbors) if n_neighbors <5000 else "knn"
        c.execute('DROP TABLE IF EXISTS {0}'.format(table_name))
        c.execute('CREATE TABLE {0} (id integer PRIMARY KEY NOT NULL, neighbors text NOT NULL)'.format(table_name))

        filename, file_extension = os.path.splitext(infile)
        if file_extension == '.mat':
            obj = loadmat(infile)
            knn = obj['knn_result']
            for idx, neighbor_list in enumerate(knn):
                line = ''
                for x in neighbor_list[:n_neighbors]:
                    line += str(x) + ' '
                command = "INSERT INTO {2} VALUES ({0}, '{1}')".format(idx, line, table_name)
                c.execute(command)
        else: #.test
            with open(infile, 'r') as knn:
                for idx, line in enumerate(knn):
                    line = ' '.join(line.split()[:n_neighbors]) if n_neighbors < 5000 else line
                    command = "INSERT INTO {2} VALUES ({0}, '{1}')".format(idx, line, table_name)
                    c.execute(command)

        conn.commit()
        conn.close()

    # def fill_db_train_points(self, infile):
    #     obj = loadmat(infile)
    #     train_points = obj['train_points']
    #     train_classes = obj['train_classes'][0]
    #
    #     conn = sqlite3.connect('../db/train_{0}.db'.format(infile))
    #     c = conn.cursor()
    #     c.execute('DROP TABLE train_set')
    #     c.execute('''CREATE TABLE train_set
    #         (id integer PRIMARY KEY NOT NULL, class integer NOT NULL, vector text NOT NULL)''')
    #     for idx, line in enumerate(train_points):
    #         command = "INSERT INTO train_set VALUES ({0}, {1}, '{2}')".format(idx, , line)
    #         c.execute(command)
    #     conn.commit()
    #     conn.close()

    def query_knn(self, dbfile, point_id, n_neighbors=5000):
        table_name = "knn_{0}".format(n_neighbors) if n_neighbors <5000 else "knn"
        self.conn = sqlite3.connect(dbfile)
        c = self.conn.cursor()
        c.execute('SELECT neighbors FROM {1} WHERE id = {0}'.format(point_id, table_name))
        return c.fetchone()

if __name__ == '__main__':
    logging.basicConfig(filename='../log/db.log',level=logging.DEBUG, format='%(asctime)s %(message)s')
    # knn_file = "../resource/knn/5000nn_result.txt"
    knn_file = '../resource/knn/knn.mat'
    manager = DbManager()
    start = time.time()
    manager.fill_db_knn('../db/annotated.db', knn_file, n_neighbors=200)
    end = time.time()
    logging.info("Filling data from file to db finished in {0}s".format(end - start))
