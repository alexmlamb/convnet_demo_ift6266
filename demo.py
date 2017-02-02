    
from loss import crossent, accuracy
import gzip
import cPickle as pickle
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
import lasagne

def init_params():

    params = {}
    scale = 0.01
    params['W1'] = theano.shared(scale * rng.normal(size = (784,512)).astype('float32'))
    params['W2'] = theano.shared(scale * rng.normal(size = (512,512)).astype('float32'))
    params['W3'] = theano.shared(scale * rng.normal(size = (512,10)).astype('float32'))

    return params

lrelu = (lambda inp: T.nnet.relu(inp, alpha = 0.02))

def network(p,x,true_y):

    x = x.flatten(2)

    h1 = lrelu(T.dot(x,p['W1']))
    h2 = lrelu(T.dot(h1, p['W2']))
    y = T.nnet.softmax(T.dot(h2, p['W3']))

    loss = crossent(y,true_y)
    acc = accuracy(y,true_y)

    return loss, acc

if __name__ == "__main__":

    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid

    trainx = trainx.reshape((50000,1,28,28)).astype('float32')
    validx = validx.reshape((10000,1,28,28)).astype('float32')

    trainy = trainy.astype('int32')
    validy = validy.astype('int32')

    print trainx.shape
    print validx.shape

    x = T.tensor4()
    y = T.ivector()

    params = init_params()

    loss, acc = network(params,x,y)

    updates = lasagne.updates.adam(loss, params.values())

    train_method = theano.function(inputs = [x,y], outputs = [loss,acc],updates=updates)
    valid_method = theano.function(inputs = [x,y], outputs = [loss,acc])

    for i in range(0,1000):
        print train_method(trainx[0:64], trainy[0:64])

