import lasagne
import theano.tensor as T
import theano,scipy
from numpy.linalg import norm
from lasagne.nonlinearities import softmax,very_leaky_rectify,softplus,sigmoid
from lasagne.layers import InputLayer, DenseLayer, get_output,MaxPool2DLayer,Conv2DLayer,Layer
from lasagne.updates import sgd, apply_momentum,apply_nesterov_momentum
from lasagne.objectives import categorical_accuracy
from lasagne.regularization import regularize_layer_params
from scipy.misc import imshow
import pickle
import theano.typed_list
import numpy as np
import random
path  =r'/media/pan/Acer/Users/Pan/Desktop/plankton/'


'''
xtrain1 = pickle.load(open(path+ r'xtrain1.p','rb'))
ytrain1 = pickle.load(open(path+ r'ytrain1.p','rb'))

n = len(ytrain1)
index = range(n)
random.shuffle(index)
#xtrain = np.array(i.ravel()/255. for i in xtrain1[0:60000]])
xtrain =[]
for i in range(60000):
    xtrain.append( xtrain1[index[i]].ravel())

ytrain = np.array(ytrain1)[index[0:60000]]
pickle.dump(xtrain,open(path+'xtrain2.p','wb'))   
pickle.dump(ytrain,open(path+'ytrain2.p','wb'))

#xtest = np.array(i.ravel()/255. for i in xtrain1[50000:n])
xtest = []
for i in range(n-60000):
    xtest.append(xtrain1[index[i+60000]].ravel())

ytest = np.array(ytrain1)[index[60000:n]]
pickle.dump(xtest,open(path+'xtest2.p','wb'))
pickle.dump(ytest,open(path+'ytest2.p','wb'))
'''
class cyclicslice(Layer):
    def get_output_for(self,p1,**kwargs):
        self.shape=p1.shape
#        pp=T.tensor4('pp')
        def rotate(p):
            r1=p.dimshuffle(0,2,1)[:,:,::-1]
            r2=p[:,:,::-1].dimshuffle(0,2,1)
            r3=p[:,::-1,:][:,:,::-1]
            r4=p
            return T.concatenate((r4,r1,r3,r2))
        tmp1,update=theano.scan(rotate,outputs_info=None,sequences=p1)
        return tmp1.reshape((tmp1.shape[0]*tmp1.shape[1],1,tmp1.shape[2],tmp1.shape[3]))
    def get_output_shape_for(self, input_shape):
        return (None,input_shape[1],input_shape[2],input_shape[3])


class cyclicpool(Layer):
    def get_output_for(self,input,**kwargs):
        tmp=input.reshape((input.shape[0]/4,4,input.shape[1]))
        return T.mean(tmp,axis=1)
    def get_output_shape_for(self, input_shape):
        return (None,input_shape[1])

rng=np.random
x1=T.matrix('x1',dtype='float32')
y1=T.vector('y1',dtype='int64')
batchsize=64
cycle=True

train_x = pickle.load(open(path+'xtrain2.p','rb'))
train_y = pickle.load(open(path+'ytrain2.p','rb'))
xtest = pickle.load(open(path+'xtest2.p','rb'))
ytest =  pickle.load(open(path+'ytest2.p','rb'))
#train_x=pickle.load(open('/home/ziheng/Desktop/cnn/train_x.pkl','r')).reshape((65000,10000))/255.0
#train_y=pickle.load(open('/home/ziheng/Desktop/cnn/train_y.pkl','r'))-1

#test_x=pickle.load(open('/home/ziheng/Desktop/cnn/test_x.pkl','r')).reshape((10000,10000))/255.0
#test_y=pickle.load(open('/home/ziheng/Desktop/cnn/test_y.pkl','r'))-1

l0=InputLayer(shape=(None,1,100,100),input_var=x1.reshape((x1.shape[0],1,100,100)))
l0_5=cyclicslice(l0)
l1=Conv2DLayer(l0_5,32,(5,5),nonlinearity=very_leaky_rectify,pad = 'same')
l2=MaxPool2DLayer(l1,(2,2))
l3=Conv2DLayer(l2,48,(3,3),nonlinearity=very_leaky_rectify,pad = 'same')
l4 = Conv2DLayer(l3,64,(3,3),nonlinearity=very_leaky_rectify,pad = 'same')
l5 = Conv2DLayer(l4,80,(3,3),nonlinearity=very_leaky_rectify,pad = 'same')
l6 = MaxPool2DLayer(l5,(2,2))
l7 = Conv2DLayer(l6,128,(3,3),nonlinearity=very_leaky_rectify,pad = 'same')
l8 = MaxPool2DLayer(l7,(2,2))
l9 = DenseLayer(l8,512,nonlinearity=very_leaky_rectify)
l10 = cyclicpool(l9)
l11=DenseLayer(l10,121,nonlinearity=softmax)

rate=theano.shared(np.cast['float32'](0.02))
params = lasagne.layers.get_all_params(l11)
prediction = lasagne.layers.get_output(l11)
l2_penalty = regularize_layer_params([l7,l5,l4,l3,l9,l11,l1], lasagne.regularization.l2)*np.cast['float32'](.0003)
loss = lasagne.objectives.categorical_crossentropy(prediction, y1)  
loss = loss.mean()+l2_penalty
updates_sgd = sgd(loss, params, learning_rate=rate)
updates = apply_nesterov_momentum(updates_sgd, params, momentum=0.8)
train_model = theano.function([x1,y1],outputs=loss,updates=updates,allow_input_downcast=True)
fprediction = theano.function([x1],outputs=prediction,allow_input_downcast=True)
pred=theano.function([x1,y1],outputs=categorical_accuracy(prediction,y1),allow_input_downcast=True)
sp_output = updates.values().append(loss)
sp_return = theano.function(inputs=[x1,y1],outputs=updates.values(),allow_input_downcast=True)
lossf = theano.function([x1,y1],outputs = loss,allow_input_downcast=True)
def spupdate(outputs,updates):
    for i in range(len(outputs)):
        updates.keys()[i].set_value(outputs[i])

train_x = np.array(train_x)
train_y = np.array(train_y)-1
xtest =np.array(xtest)
ytest = np.array(ytest)-1
n = len(train_y)
###begin to train
renewtrain=len(train_x)/batchsize
def getpvalue(params):
    vpara = []
    for i in params:
        vpara.append(i.get_value())
    return vpara

def saveparam(paramvalue,i):
    pickle.dump(paramvalue,open(path+'param_'+str(i)+'.p','wb'))

def setpvalue(para0,params):
    for i in range(len(params)):
        params[i].set_value(para0[i])

#renewtest=len(test_x)/batchsize
LOSS = []
PRED = []
i= 8892
rate.set_value(np.cast['float32'](0.0008))
LOSS = pickle.load(open(path +'LOSS_'+str(i)+'.p','rb'))
PRED = pickle.load(open(path +'PRED_'+str(i)+'.p','rb'))
#pickle.dump(LOSS,open(path +'LOSS_'+str(i)+'.p','wb'))
#pickle.dump(PRED,open(path +'PRED_'+str(i)+'.p','wb'))
para0 = pickle.load(open(path +'param_'+str(i)+'.p','rb'))
setpvalue(para0,params)
while i <=100000:
    if i%renewtrain+1 == 0:    
        tem = random.shuffle(range(n))
        train_x = train_x[tem]
        train_y = train_y[tem]
    if cycle:
        i1=i%renewtrain
        tindex=range(i1*batchsize,(i1+1)*batchsize)
        #spupdate(sp_return(train_x[tindex]/255.,train_y[tindex]),updates)        
        #newloss= lossf(train_x[tindex]/255.,train_y[tindex])
        newloss = train_model(train_x[tindex]/255.-0.9,train_y[tindex])
        LOSS.append(newloss)
        if len(LOSS) >2:
            if (LOSS[-1]-LOSS[-2])>1.5:
                print 'diverge!!'
                saveparam(para0,i)
                setpvalue(para0,params)
                rate.set_value(rate.get_value()*np.cast['float32'](0.5))    
                continue
    if i%(renewtrain/4) ==0 and i!=0:
        norm_w = sum(map(lambda a: norm(params[-a].get_value()), [1,2,3,4]))
        norm_u = sum(map(lambda a: norm(params[-a].get_value()-para0[-a]), range(1,5)))
        ratio = (-np.mean(np.array(LOSS[len(LOSS)-500:len(LOSS)]))+np.mean(np.array(LOSS[len(LOSS)-1000:len(LOSS)-500])))/np.mean(np.array(LOSS[len(LOSS)-1000:len(LOSS)-500]))
        if norm_u/norm_w<0.0005:
            pass
            #rate.set_value(rate.get_value()*np.cast['float32'](1.5))
        elif norm_u/norm_w > 0.00125 or ratio <0.075:
            rate.set_value(rate.get_value()*np.cast['float32'](0.5))
        print 'norm_W: %f, norm_u: %f,ratio: %f,learning_rate: %f' %(norm_w,norm_u,norm_u/norm_w,rate.get_value())
        temp = range(len(ytest))
        random.shuffle(temp)
        prederror1 = pred(xtest[temp[0:500]]/255.-0.9,ytest[temp[0:500]])
        prederror2 = pred(xtest[temp[500:1000]]/255.-0.9,ytest[temp[500:1000]])
        prederror3 = pred(xtest[temp[1000:1500]]/255.-0.9,ytest[temp[1000:1500]])        
        errorrate = sum(prederror1+prederror2+prederror3)/1500.
        PRED.append(errorrate)
        print '++++++++++PREDERROR:%f' %errorrate     
        pickle.dump(LOSS,open(path +'LOSS_'+str(i)+'.p','wb'))
        pickle.dump(PRED,open(path +'PRED_'+str(i)+'.p','wb'))
        saveparam(para0,i)
    if i>0:
        para0=getpvalue(params)
    print 'in %d round, the loss function is %f'%(i+1,newloss)    
    i+=1

