# How many values of \lambda to search
GRIDSIZE=131

# Number of replicates to run with fixed optimal hyperparameters to avoid
# bad random initialization
n=100

import sys, os

# Run number/random seed for data splitting and batching
RUN = 1126

# Limit to a specific GPU
DEV = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=str(DEV)

# Where to save results
outdir = 'results/'

import tensorflow
# If using tensorflow v1, proceed as normal
if int(tensorflow.__version__[0])<2:
    tf = tensorflow
# If using v2, use the compat module and disable tf2 behavior
else:
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()

import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.stats as stats
import itertools
from tqdm import tqdm
from collections import Counter

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.metrics import r2_score, roc_auc_score, log_loss

from attributionpriors.ops import AttributionPriorExplainer as TFOpsExplainer


# ## Load Data
X = pd.read_csv("X_sparsity.csv",index_col=0)#np.random.randn(10000,10)#
y = np.load("y_sparsity.npy").astype(np.int32)#X[:,1]**2+X[:,5]>0.9#

Xtv,Xtest, ytv, ytest = train_test_split(X,y,random_state=200)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xtv,ytv,random_state=100)

np.random.seed(RUN)
train_inds = np.random.choice(ytrain.shape[0],100)
valid_inds = np.random.choice(yvalid.shape[0],100)
Xtrain, ytrain = Xtrain.iloc[train_inds], ytrain[train_inds]
Xvalid, yvalid = Xvalid.iloc[valid_inds], yvalid[valid_inds]

# Feature at an index
def f(s):
    return np.where(Xtrain.columns==s)[0][0]

# Preprocess
imp = Imputer()
ss = StandardScaler()
Xtrain_imp = imp.fit_transform(Xtrain)
Xvalid_imp = imp.transform(Xvalid)
Xtest_imp = imp.transform(Xtest)
Xtrain_ss = ss.fit_transform(Xtrain_imp)
Xvalid_ss = ss.transform(Xvalid_imp)
Xtest_ss = ss.transform(Xtest_imp)

Xtrain_ss = np.clip(Xtrain_ss,-3,3)
Xvalid_ss = np.clip(Xvalid_ss,-3,3)
Xtest_ss = np.clip(Xtest_ss,-3,3)

Xtest_holdout = Xtest_ss
Xtest_ss = Xvalid_ss
ytest_holdout = ytest
ytest = yvalid

d = X.shape[1]


# ## Tensorflow Dataset

# Dataset API
def make_dataset(X, Y, batch_size, shuffle=True, buffer_size=100):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset
def sqfactors(n):
    i = int(np.sqrt(n))
    while n%i!=0:
        i -= 1
    return i, int(n/i)

# Float64
dtype = tf.float64

# Initialize datasets
references_per_batch, dset_batch_size = 1,100
ref_repeats = 100
references_per_batch *= ref_repeats
train_set = make_dataset(Xtrain_ss,np.vstack((ytrain,1-ytrain)).T.astype(np.float64),batch_size=dset_batch_size,shuffle=True,buffer_size=800)
train_set = train_set.repeat()
valid_set = make_dataset(Xvalid_ss,np.vstack((yvalid,1-yvalid)).T.astype(np.float64),batch_size=Xvalid_ss.shape[0],shuffle=True)
test_set = make_dataset(Xtest_ss,np.vstack((ytest,1-ytest)).T.astype(np.float64),batch_size=Xtest_ss.shape[0],shuffle=False)

ones_set = make_dataset(Xtrain_ss, np.vstack((np.ones_like(ytrain),np.zeros_like(ytrain))).T.astype(np.float64), batch_size=dset_batch_size, shuffle=True, buffer_size=800)
ones_set = ones_set.repeat()

# Get dataset handle
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, train_set.output_types, train_set.output_shapes)
x_pl, y_true = iterator.get_next()

# Iterators
train_iter = train_set.make_initializable_iterator()
valid_iter  = valid_set.make_initializable_iterator()
test_iter  = test_set.make_initializable_iterator()
ones_iter = ones_set.make_initializable_iterator()

# GPU options (allow growth)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# Dataset handles
train_handle = sess.run(train_iter.string_handle())
valid_handle  = sess.run(valid_iter.string_handle())
test_handle  = sess.run(test_iter.string_handle())
ones_handle  = sess.run(ones_iter.string_handle())

ref_ss = Xtrain_ss
reference_dataset = tf.data.Dataset.from_tensor_slices((np.repeat(ref_ss,ref_repeats,axis=0)))
reference_dataset = reference_dataset.shuffle(1000)
reference_dataset = reference_dataset.batch(dset_batch_size * references_per_batch)
reference_dataset = reference_dataset.repeat()
reference_iter    = reference_dataset.make_one_shot_iterator()
background_batch = reference_iter.get_next()
background_reference_op = tf.reshape(background_batch, [-1, references_per_batch, d])

# Shorthand for common data 
X, y, R = x_pl, y_true, background_reference_op

# Names for indices into results tuple, ie t[EG] gives you the EG values
PRED, EG, COST, OPT, EGOPT, EGFLAG, EXPLAINER, EGLMBD, L1LMBD, DROPLMBD = range(10)


# EG regularized with Gini
def egmodel(X,R,layers=[512],eg=None,l1=None,dropout=None):
    layers = [l for l in layers]
    eg_strength = eg if eg else 0.0
    l1_strength = tf.constant(l1,dtype=tf.float32) if l1 else tf.constant(0.0,dtype=tf.float32)
    dropout_strength = dropout if dropout else 0.0
    
    explainer = TFOpsExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(X,lambda: R)
    if len(layers)>0:
        firstlayer = layers.pop(0)
        hid_layer = tf.layers.Dense(firstlayer,activation=tf.nn.relu)
        hid = hid_layer(cond_input_op)
        if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        for l in layers:
            hid = tf.layers.Dense(l,activation=tf.nn.relu)(hid)
            if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        output = tf.layers.Dense(2)(hid)

    else:
        output_layer = tf.layers.Dense(2)
        output = output_layer(cond_input_op)
    y_pred = tf.nn.sigmoid(output)
    eg_op = explainer.shap_value_op(output, cond_input_op, y[:,1])
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y)
    total_cost = tf.reduce_mean(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(total_cost)
    
    abs_eg = tf.abs(tf.cast(eg_op,tf.float32))
    
    l1_reg = tf.reduce_mean(abs_eg) * l1_strength
    
    weighted_abs_loss = tf.reshape(tf.reduce_mean(abs_eg,axis=0),(1,-1))
    mad_loss = tf.abs(tf.transpose(weighted_abs_loss)-weighted_abs_loss)
    rmad_loss = tf.reduce_mean(mad_loss)/tf.reduce_mean(weighted_abs_loss)
    
    reg_lambda = tf.constant(eg_strength,dtype=tf.float32)
    reg_loss_op = -rmad_loss * reg_lambda
    reg_loss_op = reg_loss_op + tf.cast(l1_reg,dtype=tf.float32)

    train_eg_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(reg_loss_op)
    return (y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength)

# L1 on gradient explanations (as in Ross); also flag for gradxinp (works well but not previously used as an attribution prior)
def l1grad(X,R,layers=[512],eg=None,l1=None,dropout=None,inputs=False):
    layers = [l for l in layers]
    eg_strength = eg if eg else 0.0
    l1_strength = tf.constant(l1,dtype=tf.float64) if l1 else tf.constant(0.0,dtype=tf.float64)
    dropout_strength = dropout if dropout else 0.0
    
    explainer = TFOpsExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(X,lambda: R)
    
    if len(layers)>0:
        firstlayer = layers.pop(0)
        hid_layer = tf.layers.Dense(firstlayer,activation=tf.nn.relu)
        hid = hid_layer(cond_input_op)
        if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        kernel = hid_layer.kernel
        for l in layers:
            hid = tf.layers.Dense(l,activation=tf.nn.relu)(hid)
            if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        output = tf.layers.Dense(2)(hid)

    else:
        output_layer = tf.layers.Dense(2)
        output = output_layer(cond_input_op)
        kernel = output_layer.kernel
    
    
    y_pred = tf.nn.sigmoid(output)
    eg_op = explainer.shap_value_op(output, cond_input_op, y[:,1])
    grads = tf.gradients(output[:,1],X) # Train with gradients but use EG for explanations (you should probably use this)
    if inputs: grads *= X
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y)
    total_cost = tf.reduce_mean(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(total_cost)
    
    l1_reg = tf.reduce_mean(tf.abs(kernel)) * l1_strength 
    rmad_loss = tf.reduce_mean(tf.abs(tf.cast(grads,tf.float32)))
    
    reg_lambda = tf.constant(eg_strength,dtype=tf.float32)
    reg_loss_op = -rmad_loss * reg_lambda
    reg_loss_op = reg_loss_op + tf.cast(l1_reg,dtype=tf.float32)

    train_eg_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(reg_loss_op)
    return (y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength)

# Gini on gradients (sorta-Ross); also flag for gradxinp (works well but not previously used as an attribution prior)
def ginigrad(X,R,layers=[512],eg=None,l1=None,dropout=None,inputs=False):
    layers = [l for l in layers]
    eg_strength = eg if eg else 0.0
    l1_strength = tf.constant(l1,dtype=tf.float64) if l1 else tf.constant(0.0,dtype=tf.float64)
    dropout_strength = dropout if dropout else 0.0
    
    explainer = TFOpsExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(X,lambda: R)
    if len(layers)>0:
        firstlayer = layers.pop(0)
        hid_layer = tf.layers.Dense(firstlayer,activation=tf.nn.relu)
        hid = hid_layer(cond_input_op)
        if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        l1_reg = tf.reduce_mean(tf.abs(hid_layer.kernel)) * l1_strength
        for l in layers:
            hid = tf.layers.Dense(l,activation=tf.nn.relu)(hid)
            if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        output = tf.layers.Dense(2)(hid)

    else:
        output_layer = tf.layers.Dense(2)
        output = output_layer(cond_input_op)
        l1_reg = tf.reduce_mean(tf.abs(output_layer.kernel)) * l1_strength
        
        
    y_pred = tf.nn.sigmoid(output)#model(cond_input_op)
    eg_op = explainer.shap_value_op(output, cond_input_op, y[:,1]) # Train with gradients but use EG for explanations (you should probably use this)
    grads = tf.gradients(output[:,1],X) # Train with gradients but use EG for explanations (you should probably use this)
    if inputs: grads *= X
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y)
    total_cost = tf.reduce_mean(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(total_cost)
    
    weighted_abs_loss = tf.reshape(tf.reduce_mean(tf.abs(tf.cast(grads,tf.float32)),axis=0),(1,-1))
    mad_loss = tf.abs(tf.transpose(weighted_abs_loss)-weighted_abs_loss)
    rmad_loss = tf.reduce_mean(mad_loss)/tf.reduce_mean(weighted_abs_loss)
    
    reg_lambda = tf.constant(eg_strength,dtype=tf.float32)
    reg_loss_op = -rmad_loss * reg_lambda
    reg_loss_op = reg_loss_op + tf.cast(l1_reg,dtype=tf.float32)

    train_eg_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(reg_loss_op)
    return (y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength)        

# Sparse Group Lasso (May 2019 version)
def sglmodel(X,R,layers=[512],eg=None,l1=None,dropout=None):
    layers = [l for l in layers]
    eg_strength = eg if eg else 0.0
    l1_strength = tf.constant(l1,dtype=tf.float64) if l1 else tf.constant(0.0,dtype=tf.float64)
    dropout_strength = dropout if dropout else 0.0
    
    explainer = TFOpsExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(X,lambda: R)
    if len(layers)>0:
        firstlayer = layers.pop(0)
        hid_layer = tf.layers.Dense(firstlayer,activation=tf.nn.relu)
        hid = hid_layer(cond_input_op)
        if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        kernel = hid_layer.kernel
        for l in layers:
            hid = tf.layers.Dense(l,activation=tf.nn.relu)(hid)
            if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        output = tf.layers.Dense(2)(hid)

    else:
        output_layer = tf.layers.Dense(2)
        output = output_layer(cond_input_op)
        kernel = output_layer.kernel
    
    
    y_pred = tf.nn.sigmoid(output)
    eg_op = explainer.shap_value_op(output, cond_input_op, y[:,1])
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y)
    total_cost = tf.reduce_mean(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(total_cost)
    
    l1_ind = tf.reduce_sum(tf.abs(kernel)) 
    l1_group = tf.reduce_sum(tf.norm(kernel,axis=0))*tf.sqrt(tf.cast(kernel.shape[0],tf.float64))
    l1_reg = (l1_ind + l1_group)* l1_strength
        
    weighted_abs_loss = tf.reshape(tf.reduce_mean(tf.abs(tf.cast(eg_op,tf.float32)),axis=0),(1,-1))
    mad_loss = tf.abs(tf.transpose(weighted_abs_loss)-weighted_abs_loss)
    rmad_loss = tf.reduce_mean(mad_loss)/tf.reduce_mean(weighted_abs_loss)
    
    reg_lambda = tf.constant(eg_strength,dtype=tf.float32)
    reg_loss_op = -rmad_loss * reg_lambda
    reg_loss_op = reg_loss_op + tf.cast(l1_reg,dtype=tf.float32)

    train_eg_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(reg_loss_op)
    return (y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength)

# SGL modified to follow Scardapane more closely
def sgl_penalty_layer(k):
    l1_ind = tf.reduce_sum(tf.abs(k))
    l1_group  = tf.reduce_sum(tf.norm(k,axis=0))*tf.sqrt(tf.cast(k.shape[0],tf.float64))
    return l1_ind+l1_group
def newsgl(X,R,layers=[512],eg=None,l1=None,dropout=None):
    layers = [l for l in layers]
    eg_strength = eg if eg else 0.0
    l1_strength = tf.constant(l1,dtype=tf.float64) if l1 else tf.constant(0.0,dtype=tf.float64)
    dropout_strength = dropout if dropout else 0.0
    
    explainer = TFOpsExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(X,lambda: R)

    if len(layers)>0:
        kernels = []
        biases = []
        firstlayer = layers.pop(0)
        hid_layer = tf.layers.Dense(firstlayer,activation=tf.nn.relu)
        hid = hid_layer(cond_input_op)
        if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        kernels.append(hid_layer.kernel)
        biases.append(hid_layer.bias)
        for l in layers:
            hid_layer = tf.layers.Dense(l,activation=tf.nn.relu)
            hid = hid_layer(hid)
            if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
            kernels.append(hid_layer.kernel)
            biases.append(hid_layer.bias)
        output = tf.layers.Dense(2)(hid)

    else:
        output_layer = tf.layers.Dense(2)
        output = output_layer(cond_input_op)
        kernels = [output_layer.kernel]
        biases = [output_layer.bias]
    
    
    kernel = kernels[0]
    y_pred = tf.nn.sigmoid(output)
    eg_op = explainer.shap_value_op(output, cond_input_op, y[:,1])
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y)
    total_cost = tf.reduce_mean(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(total_cost)
    l1_reg = (tf.reduce_sum([sgl_penalty_layer(k) for k in kernels])+
              tf.reduce_sum([2*tf.reduce_sum(tf.abs(b)) for b in biases]))*l1_strength
        
    weighted_abs_loss = tf.reshape(tf.reduce_mean(tf.abs(tf.cast(eg_op,tf.float32)),axis=0),(1,-1))
    mad_loss = tf.abs(tf.transpose(weighted_abs_loss)-weighted_abs_loss)
    rmad_loss = tf.reduce_mean(mad_loss)/tf.reduce_mean(weighted_abs_loss)
    
    reg_lambda = tf.constant(eg_strength,dtype=tf.float32)
    reg_loss_op = -rmad_loss * reg_lambda
    reg_loss_op = reg_loss_op + tf.cast(l1_reg,dtype=tf.float32)

    train_eg_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(reg_loss_op)
    return (y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength)    

# Standard L1 model, all layers
def l1model(X,R,layers=[512],eg=None,l1=None,dropout=None):
    layers = [l for l in layers]
    eg_strength = eg if eg else 0.0
    l1_strength = tf.constant(l1,dtype=tf.float64) if l1 else tf.constant(0.0,dtype=tf.float64)
    dropout_strength = dropout if dropout else 0.0
    
    explainer = TFOpsExplainer()
    cond_input_op, train_eg = explainer.input_to_samples_delta(X,lambda: R)
    
    kernels = []
    if len(layers)>0:
        firstlayer = layers.pop(0)
        hid_layer = tf.layers.Dense(firstlayer,activation=tf.nn.relu)
        hid = hid_layer(cond_input_op)
        if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        kernel = hid_layer.kernel
        kernels.append(kernel)
        for l in layers:
            nextlayer = tf.layers.Dense(l,activation=tf.nn.relu)
            hid = nextlayer(hid)
            kernel = nextlayer.kernel
            kernels.append(kernel)
            if dropout: hid = tf.nn.dropout(hid,keep_prob=1-dropout_strength)
        output = tf.layers.Dense(2)(hid)

    else:
        output_layer = tf.layers.Dense(2)
        output = output_layer(cond_input_op)
        kernel = output_layer.kernel
        kernels.append(kernel)
    
    
    y_pred = tf.nn.sigmoid(output)
    eg_op = explainer.shap_value_op(output, cond_input_op, y[:,1])
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y)
    total_cost = tf.reduce_mean(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(total_cost)
    
    kernel_norms = tf.stack([tf.reduce_sum(tf.abs(k)) for k in kernels])
    l1_reg = (((tf.reduce_sum(kernel_norms)))/np.sum([np.prod(k.shape).value for k in kernels])) * l1_strength
        
    weighted_abs_loss = tf.reshape(tf.reduce_mean(tf.abs(tf.cast(eg_op,tf.float32)),axis=0),(1,-1))
    mad_loss = tf.abs(tf.transpose(weighted_abs_loss)-weighted_abs_loss)
    rmad_loss = tf.reduce_mean(mad_loss)/tf.reduce_mean(weighted_abs_loss)
    
    reg_lambda = tf.constant(eg_strength,dtype=tf.float32)
    reg_loss_op = -rmad_loss * reg_lambda
    reg_loss_op = reg_loss_op + tf.cast(l1_reg,dtype=tf.float32)

    train_eg_op = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(reg_loss_op)
    return (y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength)    


# Indices to training results tuple
PRED, EG, COST, OPT, EGOPT, EGFLAG, EXPLAINER = range(7)

def train(y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer, reg_lambda, l1_strength,
          egstrength=0.0,l1strength=0.0,sess=sess,rounds=1000,iterator=list):
    i = 0
    for i in iterator(range(rounds)):
        loss, _ = sess.run([total_cost, optimizer], feed_dict={handle:train_handle,reg_lambda:egstrength,l1_strength:l1strength})
        loss = 0.0
        _, eg = sess.run([train_eg_op, eg_op], feed_dict={handle: ones_handle, train_eg: True,reg_lambda:egstrength,l1_strength:l1strength})

# Evaluation functions
def fastpredict(truth,output,X,sess=sess):
    return sess.run(output,feed_dict={x_pl:X,y:np.vstack((truth,1-truth)).T})
predict=fastpredict

def get_train_shaps(y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer,*args):
    eg = []
    batch_input = []
    for j in range(int(Xtrain_ss.shape[0]/dset_batch_size)):
        eg_j, batch_input_j, sparse_labels, pred_labels = sess.run([eg_op, x_pl, y_true, y_pred], feed_dict={handle: ones_handle, train_eg: True})
        eg.append(eg_j)
        batch_input.append(batch_input_j)
    eg_shaps = np.vstack(eg)
    eg_X = np.vstack(batch_input)
    return eg_shaps, eg_X

def score_any(m,X,y,scorefunc=roc_auc_score):
    y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer = m
    preds = predict(y,y_pred,X)
    return scorefunc(y,preds[:,0])

def pred_any(m,X,y):
    y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer = m
    return predict(y,y_pred,X)

def test_pred(y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer,*args):
    testpreds = predict(ytest,y_pred,Xtest_ss)
    return testpreds[:,0]

def test_xent(y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer,*args):
    testpreds = predict(ytest,y_pred,Xtest_ss)
    return log_loss(ytest,testpreds[:,0])

def test_score(y_pred, eg_op, total_cost, optimizer, train_eg_op, train_eg, explainer,*args):
    testpreds = predict(ytest,y_pred,Xtest_ss)
    return roc_auc_score(ytest,testpreds[:,0])

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g
def shap_gini(shaps):
    gshaps = np.mean(np.abs(shaps),axis=0)
    return gini(gshaps)
perturbations = {
    'zero': lambda x: np.median(x),
    'perm': lambda x: np.random.choice(x,size=Xtest_ss.shape[0])
}


def shap_distribution_plot(shaps,normalize=False,maxind=None,**kwargs):
    gshaps = np.mean(np.abs(shaps),axis=0)
    if normalize: gshaps /= np.sum(gshaps)
    gorder = np.argsort(gshaps)[::-1]
    if maxind:
        plt.step(np.arange(gshaps.shape[0])[:maxind],gshaps[gorder][:maxind],**kwargs)
    else:
        plt.step(np.arange(gshaps.shape[0])[:maxind],gshaps[gorder][:maxind],**kwargs)
def shap_cdf_plot(shaps,**kwargs):
    gshaps = np.mean(np.abs(shaps),axis=0)
    oshaps = np.sort(gshaps)
    df = np.cumsum(oshaps.astype('float64'))/np.sum(oshaps).astype('float64')
    plt.step(np.arange(oshaps.shape[0]),df,**kwargs)


def initnew(sess=sess):
    uninitialized = list(set([n 
                     for v in sess.run(tf.report_uninitialized_variables()) 
                     for n in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=v)]))
    sess.run(tf.variables_initializer(uninitialized))


architectures = [
    [512,128,32]
]
penalties = np.logspace(-10,3,GRIDSIZE)


l1_params = [(a,lmbd) for a in architectures for lmbd in (penalties)]
eg_params = [(a,lmbd) for a in architectures for lmbd in (penalties)]
sgl_params = [(a,lmbd) for a in architectures for lmbd in (penalties)]
newsgl_params = [(a,lmbd) for a in architectures for lmbd in (penalties)]
l1grad_params = [(a,lmbd) for a in architectures for lmbd in (penalties)]
ginigrad_params = [(a,lmbd) for a in architectures for lmbd in (penalties)]


MROC, MXENT, MGINI, MSHAP, MPRED, MDATA = range(6)
def train_until(m,params,sess=sess,rounds=1000,iterator=list):
    failed = 1
    mt = m
    for i in range(10):
        if not failed: break
        try:
            sess.run(tf.global_variables_initializer())
            sess.run(train_iter.initializer)
            sess.run(ones_iter.initializer)
            train(*m,rounds=rounds,iterator=iterator,**params)
            r = test_score(*mt)
            s,d = get_train_shaps(*mt)
            x = test_xent(*mt)
            g = shap_gini(s)
            p = test_pred(*mt)
            failed = 0
            return (r,x,g,s,p,d)
        except ValueError:
            pass
    return None

# Train first models
eg_results = []
m = egmodel(X,R,[512,128,32])
for a,eg in tqdm(eg_params):
    t = train_until(m,{'egstrength':eg},rounds=20)
    eg_results.append((t))
    
sgl_results = []
m = sglmodel(X,R,[512,128,32])
for a,l1 in tqdm(sgl_params):
    t = train_until(m,{'l1strength':l1},rounds=20)
    sgl_results.append((t))
    
l1_results = []
m = l1model(X,R,[512,128,32])
for a,l1 in tqdm(l1_params):
    t = train_until(m,{'l1strength':l1},rounds=20)
    l1_results.append((t))
    
newsgl_results = []
m = newsgl(X,R,[512,128,32])
for a,l1 in tqdm(newsgl_params):
    t = train_until(m,{'l1strength':l1},rounds=20)
    newsgl_results.append((t))
    
l1grad_results = []
m = l1grad(X,R,[512,128,32])
for a,eg in tqdm(l1grad_params):
    t = train_until(m,{'egstrength':eg},rounds=20)
    l1grad_results.append((t))
    
ginigrad_results = []
m = ginigrad(X,R,[512,128,32])
for a,eg in tqdm(ginigrad_params):
    t = train_until(m,{'egstrength':eg},rounds=20)
    ginigrad_results.append((t))

# Find best hyperparameters
eg_scores = np.array([t[MROC] if t is not None else float('nan') for t in eg_results])
eg_ind=np.where(eg_scores==np.nanmax(eg_scores))[0][0]
    
ginigrad_scores = np.array([t[MROC] if t is not None else float('nan') for t in ginigrad_results])
ginigrad_ind=np.where(ginigrad_scores==np.nanmax(ginigrad_scores))[0][0]

l1grad_scores = np.array([t[MROC] if t is not None else float('nan') for t in l1grad_results])
l1grad_ind=np.where(l1grad_scores==np.nanmax(l1grad_scores))[0][0]
    
sgl_scores = np.array([t[MROC] if t is not None else float('nan') for t in sgl_results])
sgl_ind=np.where(sgl_scores==np.nanmax(sgl_scores))[0][0]
    
newsgl_scores = np.array([t[MROC] if t is not None else float('nan') for t in newsgl_results])
newsgl_ind=np.where(newsgl_scores==np.nanmax(newsgl_scores))[0][0]
    
l1_scores = np.array([t[MROC] if t is not None else float('nan') for t in l1_results])
l1_ind=np.where(l1_scores==np.nanmax(l1_scores))[0][0]
    

# Test performance
eg_final_models=[]
eg_final_preds=[]
m = egmodel(X,R,[512,128,32])
for a,eg in tqdm(itertools.repeat(eg_params[eg_ind],n),total=n):
    t = train_until(m,{'egstrength':eg},rounds=20)
    eg_final_models.append((t))
    eg_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))
    
ginigrad_final_models=[]
ginigrad_final_preds=[]
m = ginigrad(X,R,[512,128,32])
for a,eg in tqdm(itertools.repeat(ginigrad_params[ginigrad_ind],n),total=n):
    t = train_until(m,{'egstrength':eg},rounds=20)
    ginigrad_final_models.append((t))
    ginigrad_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))
    
l1grad_final_models=[]
l1grad_final_preds=[]
m = l1grad(X,R,[512,128,32])
for a,eg in tqdm(itertools.repeat(l1grad_params[l1grad_ind],n),total=n):
    t = train_until(m,{'egstrength':eg},rounds=20)
    l1grad_final_models.append((t))
    l1grad_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))

l1_final_models=[]
l1_final_preds=[]
m = l1model(X,R,[512,128,32])
for a,l1 in tqdm(itertools.repeat(l1_params[l1_ind],n),total=n):
    t = train_until(m,{'l1strength':l1},rounds=20)
    l1_final_models.append((t))
    l1_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))

sgl_final_models=[]
sgl_final_preds=[]
m = sglmodel(X,R,[512,128,32])
for a,l1 in tqdm(itertools.repeat(sgl_params[sgl_ind],n),total=n):
    t = train_until(m,{'l1strength':l1},rounds=20)
    sgl_final_models.append((t))
    sgl_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))

newsgl_final_models=[]
newsgl_final_preds=[]
m = newsgl(X,R,[512,128,32])
for a,l1 in tqdm(itertools.repeat(newsgl_params[newsgl_ind],n),total=n):
    t = train_until(m,{'l1strength':l1},rounds=20)
    newsgl_final_models.append((t))
    newsgl_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))

unreg_final_models=[]
unreg_final_preds=[]
m = l1model(X,R,[512,128,32])
for a in tqdm(range(n)):
    t = train_until(m,{},rounds=20)
    unreg_final_models.append((t))
    unreg_final_preds.append(fastpredict(ytest_holdout,m[0],Xtest_holdout))


# Compile results
unreg_results = []

for vresults, tresults, tpreds, name in zip(
    [eg_results, ginigrad_results, l1grad_results, l1_results, sgl_results, newsgl_results, unreg_results],
    [eg_final_models, ginigrad_final_models, l1grad_final_models, l1_final_models, sgl_final_models, newsgl_final_models, unreg_final_models],
    [eg_final_preds, ginigrad_final_preds, l1grad_final_preds, l1_final_preds, sgl_final_preds, newsgl_final_preds, unreg_final_preds],
    ['gini','ginigrad','l1grad','l1','sgl','newsgl','unreg']):

    vout = pd.DataFrame(np.zeros((len(vresults),3)),columns=['gini','roc','xent'])
    for i,t in enumerate(vresults):
        vout.iloc[i]['gini'] = t[MGINI] if t is not None else float('nan')
        vout.iloc[i]['roc'] = t[MROC] if t is not None else float('nan')
        vout.iloc[i]['xent'] = t[MXENT] if t is not None else float('nan')
    vout.to_csv("%s/results-validation-%s-%d.csv"%(outdir,name,RUN))
    tout = pd.DataFrame(np.zeros((len(tresults),6)),columns=['gini','roc','xent','finalroc','sparsezero','sparseperm'])
    for i,(t,p) in enumerate(zip(tresults,tpreds)):
        tout.iloc[i]['gini'] = t[MGINI] if t is not None else float('nan')
        tout.iloc[i]['roc'] = t[MROC] if t is not None else float('nan')
        tout.iloc[i]['xent'] = t[MXENT] if t is not None else float('nan')
        tout.iloc[i]['finalroc'] = roc_auc_score(ytest_holdout,p[:,0])
    tout.to_csv("%s/results-test-%s-%d.csv"%(outdir,name,RUN))
    sout = pd.DataFrame(np.zeros((len(tresults),tresults[0][MSHAP].shape[1])))
    for i,(t,p) in enumerate(zip(tresults,tpreds)):
        sout.iloc[i] = np.mean(np.abs(t[MSHAP]),axis=0) if t is not None else float('nan')
    sout.to_csv("%s/results-testshap-%s-%d.csv"%(outdir,name,RUN))