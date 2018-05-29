from pyspark import SparkContext, SparkConf
import numpy as np
import keras
from keras.models import Sequential
import keras.layers as ll
import pyarrow as pa
from keras.models import model_from_json
import cPickle as pickle
import time
from datetime import datetime
import struct
from argparse import ArgumentParser
import os

class mnist_loader(object):
    def __init__(self, path, scale):
        self.trainingimagesfile = os.path.join(path, "/train-images-idx3-ubyte")
        self.traininglabelsfile = os.path.join(path, "/train-labels-idx1-ubyte")
        if scale > 1:
            self.trainingimagesfile = "{0}-{1}".format(self.trainingimagesfile, scale)
            self.traininglabelsfile = '{0}-{1}'.format(self.traininglabelsfile, scale)
        self.testimagesfile = os.path.join(path, '/t10k-images-idx3-ubyte')
        self.testlabelsfile = os.path.join(path, '/t10k-labels-idx1-ubyte')

        print "Using:\n\ttraining images {0}\n\ttraining labels {1}\n\ttest images {2}\n\ttest labels {3}\n".format(self.trainingimagesfile, self.traininglabelsfile, self.testimagesfile, self.testlabelsfile)
        
    
    def load(self, idx, use_data_caching):
        if use_data_caching==1:
            startidx = idx[0]
            endidx = idx[1]
            filename = '/tmp/preprocessedmnist_{0}_{1}.dat'.format(startidx, endidx)
            try:
                f = open(filename, "r")
                X_train = pickle.load(f)
                y_train = pickle.load(f)
                X_test  = pickle.load(f)
                y_test  = pickle.load(f)
                f.close()
                print "Loaded from cache"
            except IOError as e:
                X_train, y_train, X_test, y_test = self.loadFromOriginalFiles(idx)
                f = open(filename, "w")
                pickle.dump(X_train, f)
                pickle.dump(y_train, f)
                pickle.dump(X_test, f)
                pickle.dump(y_test, f)
                f.close()
                print "Loaded from file"
        else:
            X_train, y_train, X_test, y_test = self.loadFromOriginalFiles(idx)
            print "Loaded from file"

        return(X_train, y_train, X_test, y_test)


    def loadFromOriginalFiles(self, idx):
        startidx = idx[0]
        endidx = idx[1]
        print("Loading files from idx ",startidx," to idx ",endidx,"...")
        t0 = time.time()
        fs = pa.hdfs.connect()
        t1 = time.time()
        print "Connection time: {0:.4f}".format(t1-t0)

        t0 = time.time()
        f = fs.open(self.trainingimagesfile)
        f.seek(startidx*28*28+16)
        t1 = time.time()
        print "Open and seek time: {0:.4f}".format(t1-t0)
        t0 = time.time()
        trainraw = f.read((endidx - startidx) * 28*28)
        t1 = time.time()
        print "Read time: {0:.4f}".format(t1-t0)
        f.close()
        trainbytes = bytearray(trainraw)
        
        t0 = time.time()
        t = []
        for i in xrange(0, endidx - startidx):
            subarray = np.array([trainbytes[i*28*28:(i+1)*28*28]]).reshape(28,28)
            subarray = subarray / 255.0
            t.append(subarray)
        X_train = np.array(t)
        t1 = time.time()
        print "Parsing X_train time: {0:.4f}".format(t1-t0)
        
        f = fs.open(self.traininglabelsfile)
        f.seek(startidx+8)
        trainraw = f.read(endidx - startidx)
        f.close()
        trainbytes = bytearray(trainraw)
        
        l = []
        for i in xrange(0, endidx - startidx):
            subarray = np.zeros(10).reshape(10)
            subarray[trainbytes[i]] = 1.0
            l.append(subarray)
        y_train = np.array(l)
        
        
        
        trainraw = fs.cat(self.testimagesfile)
        trainbytes = bytearray(trainraw)
        
        t = []
        for i in xrange(0, 10000):
            subarray = np.array([trainbytes[i*28*28+16:(i+1)*28*28+16]]).reshape(28,28)
            subarray = subarray / 255.0
            t.append(subarray)
        X_test = np.array(t)
        
        trainraw = fs.cat(self.testlabelsfile)
        trainbytes = bytearray(trainraw)
        
        l = []
        for i in xrange(0, 10000):
            subarray = np.zeros(10).reshape(10)
            subarray[trainbytes[i+8]] = 1.0
            l.append(subarray)
        y_test = np.array(l)
        
        return(X_train, y_train, X_test, y_test)

    def readNrSamples(self):
        fs = pa.hdfs.connect()

        f = fs.open(self.trainingimagesfile)
        f.seek(4)
        nr_samples = struct.unpack('>i', f.read(4))[0]
        print "\n\n\n\nNr samples = ",nr_samples
        f.close()
        return nr_samples

def runNetwork(idx, weights, model_json, use_data_caching, path, scale):
    loader = mnist_loader(path, scale)
    X_train, y_train, X_test, y_test = loader.load(idx, use_data_caching)

    #the network
    model = model_from_json(model_json)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    
    model.set_weights(weights)

    model.fit(X_train, y_train,
          validation_data=(X_test, y_test), epochs=1);
    
    return model.get_weights();


if __name__ == "__main__":

  # parse command line arguments
  parser = ArgumentParser()
  parser.add_argument("-p", "--path", dest="path",
                      help="Path to train and test images and labels on HDFS",
                      required=True)
  parser.add_argument("-s", "--scale", dest="scale", default=1,
                      help="Dataset scaling factor", required=False)
  parser.add_argument("-c", "--caching", dest="caching", default=0,
                      help="Caching strategy: 0 = disabled (default), 1 = enabled",
                      required=False)
  parser.add_argument("-t", "--tasks", dest="nr_tasks", type=int, default=4,
                      help="Nr of Spark tasks (default = 4)",
                      required=False)

  args = parser.parse_args()


  # create Spark context with Spark configuration
  conf = SparkConf().setAppName("Spark Count")
  sc = SparkContext(conf=conf)

  loader = mnist_loader(args.path, args.scale)
  X_train, y_train, X_test, y_test = loader.loadFromOriginalFiles((0,1))
  nr_samples = loader.readNrSamples()

  #the network
  model = Sequential(name="mlp")

  model.add(ll.InputLayer([28, 28]))
  model.add(ll.Flatten())
  model.add(ll.Dense(50))
  model.add(ll.Activation('sigmoid'))
  model.add(ll.Dense(10, activation='softmax'))

  model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

  model.summary()
  
  model_json = model.to_json()
  weights=model.get_weights()
    
  
  nr_partitions = args.nr_tasks
  nr_samples_partition = nr_samples / nr_partitions
  idx = []
  for i in xrange(nr_partitions):
    idx.append((i * nr_samples_partition, (i + 1) * nr_samples_partition))
  print "idx = ", idx
  idxp = sc.parallelize(idx, nr_partitions)
  
  for i in xrange(10):
    o=idxp.map(lambda x:runNetwork(x, weights, model_json,
                                   args.caching,
                                   args.path,
                                   args.scale)).collect()
    npo=np.array(o)
    print "Fertig."
  
    average_weights=np.average(npo,axis=0)
    weights = average_weights.tolist()
    
    model.set_weights(weights)
    metrics = model.evaluate(X_test, y_test)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


  print " \n\nDone with training. Summary:\n"
  for i in range(len(model.metrics_names)):
      print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

  print "\n\n"

