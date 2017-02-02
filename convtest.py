import theano
import theano.tensor as T
import numpy.random as rng

x = T.ftensor4()
w = theano.shared(rng.normal(size = (128,64,3,3)).astype('float32'))

h = T.nnet.conv.conv2d(x,w)

f = theano.function([x], h)

xi = rng.normal(size=(1,64,10,10)).astype('float32')

print f(xi).shape

