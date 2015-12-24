#use to augment training data
from scipy.ndimage.interpolation import shift,rotate,zoom
from random import randint
import scipy
from numpy.random import uniform
from skimage.transform import SimilarityTransform,warp
from numpy import exp,log
import pylab
import numpy as np



def augment(X):
    r = X
    #flip
    if randint(0,1):
        r=np.flipud(r)
    #stretch
    r = shift(r,shift=(randint(-10,10),randint(-10,10)),mode = 'constant',cval = 255)
    r = rotate(r,angle = randint(0,360),mode = 'constant',cval = 255)
    r = zoom(r,zoom = (log(uniform(exp(1/1.25),exp(1.25))),log(uniform(exp(1/1.25),exp(1.25)))),mode = 'constant',cval = 255)
    k = max(r.shape)
    w = r.shape[1]
    h = r.shape[0]
    rr = 255*np.ones(k*k,dtype = int).reshape(k,k)
    rr[(k/2-h/2):(k/2-h/2+h),(k/2-w/2):(k/2-w/2+w)] = r
    return rr
    
