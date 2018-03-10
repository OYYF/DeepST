# -*- coding: utf-8 -*-
""" 
    THEANO_FLAGS="device=gpu0" python exptTaxiBJ-L12.py
"""
from __future__ import print_function
import os
#import cPickle as pickle
import pickle
import time
import numpy as np
import h5py

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib


from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import TaxiBJ


import pandas as pds

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



TrendInterval = 0




#from ..preprocessing import MinMaxNormalization, remove_incomplete_days, timestamp2vec

class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print ('X.min(): ', X.min(), 'X.max()', X.max())
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH  # data path, you may set your own data path with a global envirmental variable DATAPATH
CACHEDATA = False  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE')  # cache path
nb_epoch = 10  # number of epoch at training stage
nb_epoch_cont = 10  # number of epoch at training (cont) stage
batch_size = 48  # batch size
T = 48  # number of time intervals at a day
lr = 0.0002  # learning rate
len_closeness = 1  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 0  # length of trend dependent sequence
nb_residual_unit = 12  # number of residual units
external_dim = 7+48


nb_flow = 2  # there are two types of flows: inflow and outflow
# split data into two subsets: Train & Test, of which the test set is the last 4 weeks
days_test = 7 * 4
len_test = T * days_test
map_height, map_width = 64, 64  # grid size
path_result = 'RET'
path_model = 'MODEL'

# if os.path.isdir(path_result) is False:
#     os.mkdir(path_result)
# if os.path.isdir(path_model) is False:
#     os.mkdir(path_model)
# if CACHEDATA and os.path.isdir(path_cache) is False:
#     os.mkdir(path_cache)

def time2vec(len_all, startWeekday, start):
    ret = []
    ret_0 = []
    vec = [(i//48 + startWeekday)%7 for i in range(start, len_all)]
    for x in vec:
        v = [0 for _ in range(7)]
        v[x] = 1
        """
        Weeked Part, to be finished
        """
        ret_0.append(v)
    ret_1 = []
    vec = [i%48 for i in range(start, len_all)]
    for x in vec:
        v = [0 for _ in range(48)]
        v[x] = 1
        ret_1.append(v)
    ret.append(ret_0)
    ret.append(ret_1)
    ret = np.hstack(ret)
    print(np.asarray(ret).shape)
    return np.asarray(ret)

def myGetData(len_closeness):
    mmnFlag = True
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    f_64=pds.read_csv('../../../../data/TaxiBJ/big_slide_spatial_feature.csv', header =None)
    array_64=f_64.values
    print ('array_64 shape: ', array_64.shape)
    print ('array_64 len: ', array_64)

    mmn = MinMaxNormalization()
    mmn.fit(array_64)
    data_all_mmn = [mmn.transform(d) for d in array_64]

    XC, XP, XT = [], [], []
    Y = []
    if mmnFlag:
        for i in range(TrendInterval*T, len(data_all_mmn)-1):
            _XC = []
            for j in range(len_closeness):
                _XC.append(np.reshape(data_all_mmn[i-j],(2,64,64))[0])
                _XC.append(np.reshape(data_all_mmn[i-j],(2,64,64))[1])
            XC.append(_XC)
            Y.append(np.reshape(data_all_mmn[i+1],(2,64,64)))
    else:
        for i in range(TrendInterval*T, len(array_64)-1):
            _XC = []
            for j in range(len_closeness):
                _XC.append(np.reshape(array_64[i-j],(2,64,64))[0])
                _XC.append(np.reshape(array_64[i-j],(2,64,64))[1])
            XC.append(_XC)
            Y.append(np.reshape(array_64[i+1],(2,64,64)))
    XC = np.asarray(XC)
    Y = np.asarray(Y)
        
    XC_train, Y_train = XC[
        :-len_test], Y[:-len_test]
    XC_test, Y_test = XC[
        -len_test:], Y[-len_test:]
    for l, X_ in zip([len_closeness], [XC_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness], [XC_test]):
        if l > 0:
            X_test.append(X_)
    
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)
    print(type(X_train), type(Y_train), type(X_test), type(Y_test))
    timeVec = time2vec(len_all=len(array_64)-1, startWeekday=5, start=TrendInterval*T)
    timeVec_train = timeVec[:-len_test]
    timeVec_test = timeVec[-len_test:]
    X_train.append(timeVec_train)
    X_test.append(timeVec_test)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)
    print(type(X_train), type(Y_train), type(X_test), type(Y_test))
    return X_train, Y_train, X_test, Y_test, mmn

# def myGetData(len_closeness, len_period=1, len_trend=1, PeriodInterval=1, TrendInterval=7):
#     mmnFlag = True
#     X_train = []
#     X_test = []
#     Y_train = []
#     Y_test = []
#     f_64=pds.read_csv('../../../../data/big_slide_spatial_feature.csv', header =None)
#     array_64=f_64.values
#     print ('array_64 shape: ', array_64.shape)
#     print ('array_64 len: ', array_64)

#     mmn = MinMaxNormalization()
#     mmn.fit(array_64)
#     data_all_mmn = [mmn.transform(d) for d in array_64]

#     XC, XP, XT = [], [], []
#     Y = []
#     if mmnFlag:
#         for i in range(TrendInterval*T, len(data_all_mmn)-1):
#             _XC = []
#             for j in range(len_closeness):
#                 _XC.append(np.reshape(data_all_mmn[i-j],(2,64,64))[0])
#                 _XC.append(np.reshape(data_all_mmn[i-j],(2,64,64))[1])
#             XC.append(_XC)
#             XP.append(np.reshape(data_all_mmn[i-PeriodInterval*T],(2,64,64)))
#             XT.append(np.reshape(data_all_mmn[i-TrendInterval*T],(2,64,64)))
#             Y.append(np.reshape(data_all_mmn[i+1],(2,64,64)))
#     else:
#         for i in range(TrendInterval*T, len(array_64)-1):
#             _XC = []
#             for j in range(len_closeness):
#                 _XC.append(np.reshape(array_64[i-j],(2,64,64))[0])
#                 _XC.append(np.reshape(array_64[i-j],(2,64,64))[1])
#             XC.append(_XC)
#             XP.append(np.reshape(array_64[i-PeriodInterval*T],(2,64,64)))
#             XT.append(np.reshape(array_64[i-TrendInterval*T],(2,64,64)))
#             Y.append(np.reshape(array_64[i+1],(2,64,64)))
#     XC = np.asarray(XC)
#     XP = np.asarray(XP)
#     XT = np.asarray(XT)
#     Y = np.asarray(Y)
        
#     XC_train, XP_train, XT_train, Y_train = XC[
#         :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
#     XC_test, XP_test, XT_test, Y_test = XC[
#         -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
#     for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
#         if l > 0:
#             X_train.append(X_)
#     for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
#         if l > 0:
#             X_test.append(X_)
    
#     print('train shape:', XC_train.shape, Y_train.shape,
#           'test shape: ', XC_test.shape, Y_test.shape)
#     print(type(X_train), type(Y_train), type(X_test), type(Y_test))
#     timeVec = time2vec(len_all=len(array_64)-1, startWeekday=5, start=TrendInterval*T)
#     timeVec_train = timeVec[:-len_test]
#     timeVec_test = timeVec[-len_test:]
#     X_train.append(timeVec_train)
#     X_test.append(timeVec_test)
#     print('train shape:', XC_train.shape, Y_train.shape,
#           'test shape: ', XC_test.shape, Y_test.shape)
#     print(type(X_train), type(Y_train), type(X_test), type(Y_test))
#     return X_train, Y_train, X_test, Y_test, mmn


def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    print (type(c_conf))
    print(c_conf)
    print (type(p_conf))
    print(p_conf)
    print (type(t_conf))
    print(t_conf)
    print('external_dim: ', external_dim)
    print('nb_residual_unit: ', nb_residual_unit)
    model = stresnet(c_conf=c_conf, p_conf=None, t_conf=None,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


# def read_cache(fname):
#     mmn = pickle.load(open('preprocessing.pkl', 'rb'))

#     f = h5py.File(fname, 'r')
#     num = int(f['num'].value)
#     X_train, Y_train, X_test, Y_test = [], [], [], []
#     for i in range(num):
#         X_train.append(f['X_train_%i' % i].value)
#         X_test.append(f['X_test_%i' % i].value)
#     Y_train = f['Y_train'].value
#     Y_test = f['Y_test'].value
#     external_dim = f['external_dim'].value
#     timestamp_train = f['T_train'].value
#     timestamp_test = f['T_test'].value
#     f.close()

#     return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


# def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
#     h5 = h5py.File(fname, 'w')
#     h5.create_dataset('num', data=len(X_train))

#     for i, data in enumerate(X_train):
#         h5.create_dataset('X_train_%i' % i, data=data)
#     # for i, data in enumerate(Y_train):
#     for i, data in enumerate(X_test):
#         h5.create_dataset('X_test_%i' % i, data=data)
#     h5.create_dataset('Y_train', data=Y_train)
#     h5.create_dataset('Y_test', data=Y_test)
#     external_dim = -1 if external_dim is None else int(external_dim)
#     h5.create_dataset('external_dim', data=external_dim)
#     h5.create_dataset('T_train', data=timestamp_train)
#     h5.create_dataset('T_test', data=timestamp_test)
#     h5.close()


def main():
    # load data
    print("loading data...")
    # ts = time.time()

    # fname = os.path.join(DATAPATH, 'CACHE', 'TaxiBJ_C{}_P{}_T{}_noExternal.h5'.format(
    #     len_closeness, len_period, len_trend))
    # if os.path.exists(fname) and CACHEDATA:
    #     X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
    #         fname)
    #     print("load %s successfully" % fname)
    # else:
    #     X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiBJ.load_data(
    #         T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
    #         preprocess_name='preprocessing.pkl', meta_data=False, meteorol_data=False, holiday_data=False)
    #     if CACHEDATA:
    #         cache(fname, X_train, Y_train, X_test, Y_test,
    #               external_dim, timestamp_train, timestamp_test)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    # print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))
    
    X_train, Y_train, X_test, Y_test, mmn = myGetData(len_closeness)

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}.noExternal'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("\nelapsed time (compiling model): %.3f seconds\n" %
          (time.time() - ts))

    print('=' * 10)
    print("training model...")
    print(len(X_train))
    for i in range(len(X_train)):
        print (len(X_train[i]))
    print(len(Y_train))
    ##X_train = X_train[:,:1372]
    ##Y_train = Y_train[:1372]
    #for i in range(len(X_train)):
    #    X_train[i] = X_train[i][:1372]
    #Y_train = Y_train[:1372]
    #ts = time.time()
    print(device_lib.list_local_devices())
    print(len(X_train))
    for x in X_train:
        print (x.shape)
    print(len(X_test))
    for x in X_test:
        print (x.shape)
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0] // 48, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    
    model.save('my_model_epoch_1.h5')
    
    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join(
        'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=2, batch_size=batch_size, callbacks=[
                        model_checkpoint])
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the final model')
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0] // 48, verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))

if __name__ == '__main__':
    main()