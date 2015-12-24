#Image_augment
import numpy as np
import pylab
import numpy
#import mahotas as mh
import pickle
import os
from scipy.misc import imresize
from augment import augment
from random import shuffle

path = r'C:\Users\Pan\Desktop\plankton/'
te_path = r'C:\Users\Pan\Desktop\plankton\test/'
tr_path = r'C:\Users\Pan\Desktop\plankton\train/'
#path = r'/media/pan/Acer/Users/Pan/Desktop/plankton/'
#tr_path = r'/media/pan/Acer/Users/Pan/Desktop/plankton/train/'
#te_path = r'/media/pan/Acer/Users/Pan/Desktop/plankton/test/'

#image read
'''
pathes = [i[0] for i in os.walk(tr_path)]
aaa =[i[1] for i in os.walk(tr_path)][0]
bbb = [i[2] for i in os.walk(tr_path)]

test_pathes = [i[0] for i in os.walk(te_path)]
ccc= [i[2] for i in os.walk(te_path)]

test = map(lambda i: mh.imread(te_path+i),ccc[0])
xtrain =[]
ytrain= []
label = {}
for i in range(1,len(pathes)):
    for j in bbb[i]:
        xtrain.append(mh.imread(pathes[i]+'/'+j))
        ytrain.append(i)
    label['aaa[i-1]'] = i

pickle.dump(xtrain,open(path+'xtrain.p','w'))
pickle.dump(ytrain,open(path+'ytrain.p','w'))
pickle.dump(test,open(path+'test.p','w'))
'''


xtrain = pickle.load(open(path+'xtrain.p','r'))
ytrain = pickle.load(open(path+'ytrain.p','r'))
xtrain1 = []
ytrain1 = []
n_class = np.zeros(121,dtype= 'int')
for i in ytrain:
    n_class[i-1]+=1

for i in range(len(xtrain)):
    if n_class[ytrain[i]-1] < 500:
        xtrain1.append(imresize(xtrain[i],(100,100)))
        ytrain1.append(ytrain[i])
        for j in range(500/n_class[ytrain[i]-1]):
            xtrain1.append(imresize(augment(xtrain[i]),(100,100)))
            ytrain1.append(ytrain[i])
    else:
        xtrain1.append(imresize(xtrain[i],(100,100)))
        ytrain1.append(ytrain[i])

        
pickle.dump(xtrain1,open(path+'xtrain1.p','wb'))
pickle.dump(ytrain1,open(path+'ytrain1.p','wb'))
'''
xtrain1 = pickle.load(open(path+'xtrain1.p','r'))
ytrain1 = pickle.load(open(path+'ytrain1.p','r'))

pickle.dump(xtrain1,open(path+'xtrain1.p','wb'))
pickle.dump(ytrain1,open(path+'ytrain1.p','wb'))
'''
