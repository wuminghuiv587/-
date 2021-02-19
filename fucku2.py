# encoding:utf-8
# 用来测试英文论文的性能
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
from sklearn.model_selection import train_test_split
import xlrd
from math import sqrt
# from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import explained_variance_score
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import roc_auc_score
import keras
import tensorflow as tf
import keras.layers.recurrent
from keras import backend as K
from sklearn import metrics
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.callbacks import TensorBoard
import time
from sklearn.metrics import roc_curve, auc, roc_auc_score  ###计算roc和auc
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from keras.callbacks import ModelCheckpoint
from keras.models import save_model
from keras.models import load_model
from keras import optimizers
from keras.layers import LSTM
# from sklearn. import decision
#from sklearn.preprocessing import Imputer


def picky(data, num):
    k = []
    for i in range(len(data)):
        if (i % num == 0):
            k.append(data[i])
    return k


#best_weights_filepath = '/root/xulun/selflstm20200427/best_weights.hdf5'
#best_weights_filepath = '/root/xulun/standardlstm0427/best_weights.hdf5'
# timestep = [5,10,15,20]
# istep = [5,10,15,20]
timestep = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
#istep = [7]
# timestep = [30][5,10,15,20,25,30]
# istep = [6][5]
# 5 没训练
# 2520
########读取训练数据

row_data = pd.read_excel('/root/xulun/316/20180514/data/trainvalue.xlsx')
#row_data = pd.read_excel('F:/FUCK/trainvalue.xlsx')
column = ['18S04Ijmb', '18S04Ts', '18S04Ftm', '18S04Fjm',
          '18S04F2k', '18S04F4k', '18S04F8k',
          '18S04UAm', '18S04UBm', '18S04UCm', '18S04UAa',
          '18S04UBa', '18S04UCa', '18S04U2k', '18S04U4k', '18S04U8k',
          '18S04V5p', '18S04V12p', '18S04V12n', '18S04Vsy15p',
          '18S04Vsy15n', '18S04Vc15p', '18S04Vc15n',
          '18S04Itma', '18S04Itmb', '18S04Itmc', '18S04Ijma', '18S04Ijmc',
          #      '18S04Zkz','18S04Ffz',
          '18S04Ncrc', '18S04Tt', '18S04Txt', '18S04Tyt', '18S04Tzt',
          '18S04Txj', '18S04Tyj', '18S04Iwk', '18S04Wxt', '18S04Wyt', '18S04Wzt', '18S04Wxj', '18S04Wyj',
          '18S04Ix', '18S04Iy','18S04Iz',
          '18S04Iy', '18S04Nw', '18S04Nwc', '18S04Ntc', '18S04Tb', '18S04Ny', '18S04N422', '18S04Zkzb0', '18S04Zkzb1',
          '18S04Zkzb2', '18S04Zkzb3', '18S04Zkzb4', '18S04Zkzb5', '18S04Zkzb6',
          '18S04Zkzb7', '18S04Zkzb8', '18S04Zkzb9', '18S04Zkzb10', '18S04Zkzb11_13',
          '18S04Zkzb14', '18S04Ffzb0', '18S04Ffzb1', '18S04Ffzb2', '18S04Ffzb3',
          '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ncrcb0',
          '18S04Ncrcb1', '18S04Ncrcb2', '18S04Ncrcb3', '18S04Ncrcb4', '18S04Ncrcb5',
          '18S04Ncrcb6', '18S04Ncrcb7',
          '18S02Fztz', '18S02Fjd', '18S02Nsa2', '18S02Fc', '18S02Fztz0', '18S02Fztz1', '18S02Fztz2',
          '18S02Fztz3', '18S02Fztz4', '18S02Fztz5', '18S02Fztz6', '18S02Fztz7', '18S02Fztz8', '18S02Fztz9',
          '18S02Fztz10', '18S02Fztz11', '18S02Fztz12', '18S02Fztz13', '18S02Fztz14', '18S02Fztz15', '18S02Fc0',
          '18S02Fc1', '18S02Fc2', '18S02Fc3', '18S02Fc4', '18S02Fc5', '18S02Fc6', '18S02Fc7', '18S02Fc8', '18S02Fc9',
          '18S02Fc10', '18S02Fc11', '18S02Fc12', '18S02Fc13', '18S02Fc14',
          'signal', '17S03Time', '17S03NDA', '17S03SJ', '17S03NP', '17S03Mxsyp', '17S03MZLsyXp',
          '17S03Mxsyn', '17S03MZLsyXn', '17S03Mysyp', '17S03MZLsyYp', '17S03Mysyn',
          '17S03MZLsyYn', '17S03Mzsyp', '17S03MZLsyZp', '17S03Mzsyn', '17S03MZLsyZn',
          '17S03Dypj', '17S03Dypc', '17S03Dxj', '17S03Dxc', '17S03Dyj', '17S03Dyc',
          '17S03Dzj', '17S03Dzc', '17S03TDY1', '17S03TDX', '17S03TDZ', '17S03TDY',
          '17S03Mxtjp', '17S03MZLtjXp', '17S03Mxtjn', '17S03MZLtjXn', '17S03Mytjp',
          '17S03MZLtjYp', '17S03Mytjn', '17S03MZLtjYn', '17S03Mx/ytjp', '17S03Ijjx', '17S03Ijjy'
    , '17S03Ijjz', '17S03MZLsyX', '17S03MZLsyY', '17S03MZLsyZ', '17S03MZLtjX', '17S03MZLtjY'
    , '17S03MZLsyX(1s)', '17S03MZLsyY(1s)', '17S03MZLsyZ(1s)', '17S03MZLtjX(1s)', '17S03MZLtjY(1s)'
          ]
value1 = row_data[column]
print('valus1', value1.shape)
row_data = pd.read_excel('/root/xulun/316/20180514/data/traindf.xlsx')
column = ['18S04Ijmb', '18S04Ts', '18S04Ftm', '18S04Fjm',
          '18S04F2k', '18S04F4k', '18S04F8k',
          '18S04UAm', '18S04UBm', '18S04UCm', '18S04UAa',
          '18S04UBa', '18S04UCa', '18S04U2k', '18S04U4k', '18S04U8k',
          '18S04V5p', '18S04V12p', '18S04V12n', '18S04Vsy15p',
          '18S04Vsy15n', '18S04Vc15p', '18S04Vc15n',
          '18S04Itma', '18S04Itmb', '18S04Itmc', '18S04Ijma', '18S04Ijmc',
          #      '18S04Zkz','18S04Ffz',
          '18S04Ncrc', '18S04Tt', '18S04Txt', '18S04Tyt', '18S04Tzt',
          '18S04Txj', '18S04Tyj', '18S04Iwk', '18S04Wxt', '18S04Wyt', '18S04Wzt', '18S04Wxj', '18S04Wyj',
          '18S04Ix', '18S04Iy','18S04Iz' ,
          '18S04Iy', '18S04Nw', '18S04Nwc', '18S04Ntc', '18S04Tb', '18S04Ny', '18S04N422', '18S04Zkzb0', '18S04Zkzb1',
          '18S04Zkzb2', '18S04Zkzb3', '18S04Zkzb4', '18S04Zkzb5', '18S04Zkzb6',
          '18S04Zkzb7', '18S04Zkzb8', '18S04Zkzb9', '18S04Zkzb10', '18S04Zkzb11_13',
          '18S04Zkzb14', '18S04Ffzb0', '18S04Ffzb1', '18S04Ffzb2', '18S04Ffzb3',
          '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ncrcb0',
          '18S04Ncrcb1', '18S04Ncrcb2', '18S04Ncrcb3', '18S04Ncrcb4', '18S04Ncrcb5',
          '18S04Ncrcb6', '18S04Ncrcb7',
          '18S02Fztz', '18S02Fjd', '18S02Nsa2', '18S02Fc', '18S02Fztz0', '18S02Fztz1', '18S02Fztz2',
          '18S02Fztz3', '18S02Fztz4', '18S02Fztz5', '18S02Fztz6', '18S02Fztz7', '18S02Fztz8', '18S02Fztz9',
          '18S02Fztz10', '18S02Fztz11', '18S02Fztz12', '18S02Fztz13', '18S02Fztz14', '18S02Fztz15', '18S02Fc0',
          '18S02Fc1', '18S02Fc2', '18S02Fc3', '18S02Fc4', '18S02Fc5', '18S02Fc6', '18S02Fc7', '18S02Fc8', '18S02Fc9',
          '18S02Fc10', '18S02Fc11', '18S02Fc12', '18S02Fc13', '18S02Fc14',
          'signal', '17S03Time', '17S03NDA', '17S03SJ', '17S03NP', '17S03Mxsyp', '17S03MZLsyXp',
          '17S03Mxsyn', '17S03MZLsyXn', '17S03Mysyp', '17S03MZLsyYp', '17S03Mysyn',
          '17S03MZLsyYn', '17S03Mzsyp', '17S03MZLsyZp', '17S03Mzsyn', '17S03MZLsyZn',
          '17S03Dypj', '17S03Dypc', '17S03Dxj', '17S03Dxc', '17S03Dyj', '17S03Dyc',
          '17S03Dzj', '17S03Dzc', '17S03TDY1', '17S03TDX', '17S03TDZ', '17S03TDY',
          '17S03Mxtjp', '17S03MZLtjXp', '17S03Mxtjn', '17S03MZLtjXn', '17S03Mytjp',
          '17S03MZLtjYp', '17S03Mytjn', '17S03MZLtjYn', '17S03Mx/ytjp', '17S03Ijjx', '17S03Ijjy'
    , '17S03Ijjz', '17S03MZLsyX', '17S03MZLsyY', '17S03MZLsyZ', '17S03MZLtjX', '17S03MZLtjY'
    , '17S03MZLsyX(1s)', '17S03MZLsyY(1s)', '17S03MZLsyZ(1s)', '17S03MZLtjX(1s)', '17S03MZLtjY(1s)'
          ]
value2 = row_data[column]
print('valus2', value2.shape)
#  value2是故障  value4是正常    训练集是故障加正常
# v2:   数据改为正常数据加故障数据  标签连接 value6=value2  value5=value4+value6  df1==values4
#  少数量样本，观察误差      valuesforward=values4   总数据为 values4+values2+values4  测试数据贯穿故障，画出roc曲线
# values6=values2[0:40000]
# trainvalue = value1.append(value2.append(value1.append(value2.append(value1[:len(value1 ) - 1118]))))
# 24W[ValueError: cannot reshape array of size 2653510 into shape (1065,15,166)]
# trainvalue = value1.append(value2.append(value1.append(value1.append(value1[:len(value1 ) - 1118 - 4444]))))
# 18w  ------20200110 10:39
trainvalue = value1.append(value2.append(value1.append(value2.append(value1.append(value1[:10876])))))
# 判断
print(trainvalue.shape)
row_data = pd.read_excel('/root/xulun/316/20180514/data/testdf.xlsx')
column = ['18S04Ijmb', '18S04Ts', '18S04Ftm', '18S04Fjm',
          '18S04F2k', '18S04F4k', '18S04F8k',
          '18S04UAm', '18S04UBm', '18S04UCm', '18S04UAa',
          '18S04UBa', '18S04UCa', '18S04U2k', '18S04U4k', '18S04U8k',
          '18S04V5p', '18S04V12p', '18S04V12n', '18S04Vsy15p',
          '18S04Vsy15n', '18S04Vc15p', '18S04Vc15n',
          '18S04Itma', '18S04Itmb', '18S04Itmc', '18S04Ijma', '18S04Ijmc',
          #      '18S04Zkz','18S04Ffz',
          '18S04Ncrc', '18S04Tt', '18S04Txt', '18S04Tyt', '18S04Tzt',
          '18S04Txj', '18S04Tyj', '18S04Iwk', '18S04Wxt', '18S04Wyt', '18S04Wzt', '18S04Wxj', '18S04Wyj',
          '18S04Ix', '18S04Iy','18S04Iz',
          '18S04Iy', '18S04Nw', '18S04Nwc', '18S04Ntc', '18S04Tb', '18S04Ny', '18S04N422', '18S04Zkzb0', '18S04Zkzb1',
          '18S04Zkzb2', '18S04Zkzb3', '18S04Zkzb4', '18S04Zkzb5', '18S04Zkzb6',
          '18S04Zkzb7', '18S04Zkzb8', '18S04Zkzb9', '18S04Zkzb10', '18S04Zkzb11_13',
          '18S04Zkzb14', '18S04Ffzb0', '18S04Ffzb1', '18S04Ffzb2', '18S04Ffzb3',
          '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ncrcb0',
          '18S04Ncrcb1', '18S04Ncrcb2', '18S04Ncrcb3', '18S04Ncrcb4', '18S04Ncrcb5',
          '18S04Ncrcb6', '18S04Ncrcb7',
          '18S02Fztz', '18S02Fjd', '18S02Nsa2', '18S02Fc', '18S02Fztz0', '18S02Fztz1', '18S02Fztz2',
          '18S02Fztz3', '18S02Fztz4', '18S02Fztz5', '18S02Fztz6', '18S02Fztz7', '18S02Fztz8', '18S02Fztz9',
          '18S02Fztz10', '18S02Fztz11', '18S02Fztz12', '18S02Fztz13', '18S02Fztz14', '18S02Fztz15', '18S02Fc0',
          '18S02Fc1', '18S02Fc2', '18S02Fc3', '18S02Fc4', '18S02Fc5', '18S02Fc6', '18S02Fc7', '18S02Fc8', '18S02Fc9',
          '18S02Fc10', '18S02Fc11', '18S02Fc12', '18S02Fc13', '18S02Fc14',
          'signal', '17S03Time', '17S03NDA', '17S03SJ', '17S03NP', '17S03Mxsyp', '17S03MZLsyXp',
          '17S03Mxsyn', '17S03MZLsyXn', '17S03Mysyp', '17S03MZLsyYp', '17S03Mysyn',
          '17S03MZLsyYn', '17S03Mzsyp', '17S03MZLsyZp', '17S03Mzsyn', '17S03MZLsyZn',
          '17S03Dypj', '17S03Dypc', '17S03Dxj', '17S03Dxc', '17S03Dyj', '17S03Dyc',
          '17S03Dzj', '17S03Dzc', '17S03TDY1', '17S03TDX', '17S03TDZ', '17S03TDY',
          '17S03Mxtjp', '17S03MZLtjXp', '17S03Mxtjn', '17S03MZLtjXn', '17S03Mytjp',
          '17S03MZLtjYp', '17S03Mytjn', '17S03MZLtjYn', '17S03Mx/ytjp', '17S03Ijjx', '17S03Ijjy'
    , '17S03Ijjz', '17S03MZLsyX', '17S03MZLsyY', '17S03MZLsyZ', '17S03MZLtjX', '17S03MZLtjY'
    , '17S03MZLsyX(1s)', '17S03MZLsyY(1s)', '17S03MZLsyZ(1s)', '17S03MZLtjX(1s)', '17S03MZLtjY(1s)'
          ]
testerror = row_data[column]
print('testerro', testerror.shape)
row_data = pd.read_excel('/root/xulun/316/20180514/data/test+value.xlsx')
column = ['18S04Ijmb', '18S04Ts', '18S04Ftm', '18S04Fjm',
          '18S04F2k', '18S04F4k', '18S04F8k',
          '18S04UAm', '18S04UBm', '18S04UCm', '18S04UAa',
          '18S04UBa', '18S04UCa', '18S04U2k', '18S04U4k', '18S04U8k',
          '18S04V5p', '18S04V12p', '18S04V12n', '18S04Vsy15p',
          '18S04Vsy15n', '18S04Vc15p', '18S04Vc15n',
          '18S04Itma', '18S04Itmb', '18S04Itmc', '18S04Ijma', '18S04Ijmc',
          #      '18S04Zkz','18S04Ffz',
          '18S04Ncrc', '18S04Tt', '18S04Txt', '18S04Tyt', '18S04Tzt',
          '18S04Txj', '18S04Tyj', '18S04Iwk', '18S04Wxt', '18S04Wyt', '18S04Wzt', '18S04Wxj', '18S04Wyj',
          '18S04Ix', '18S04Iy','18S04Iz',
          '18S04Iy', '18S04Nw', '18S04Nwc', '18S04Ntc', '18S04Tb', '18S04Ny', '18S04N422', '18S04Zkzb0', '18S04Zkzb1',
          '18S04Zkzb2', '18S04Zkzb3', '18S04Zkzb4', '18S04Zkzb5', '18S04Zkzb6',
          '18S04Zkzb7', '18S04Zkzb8', '18S04Zkzb9', '18S04Zkzb10', '18S04Zkzb11_13',
          '18S04Zkzb14', '18S04Ffzb0', '18S04Ffzb1', '18S04Ffzb2', '18S04Ffzb3',
          '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ncrcb0',
          '18S04Ncrcb1', '18S04Ncrcb2', '18S04Ncrcb3', '18S04Ncrcb4', '18S04Ncrcb5',
          '18S04Ncrcb6', '18S04Ncrcb7',
          '18S02Fztz', '18S02Fjd', '18S02Nsa2', '18S02Fc', '18S02Fztz0', '18S02Fztz1', '18S02Fztz2',
          '18S02Fztz3', '18S02Fztz4', '18S02Fztz5', '18S02Fztz6', '18S02Fztz7', '18S02Fztz8', '18S02Fztz9',
          '18S02Fztz10', '18S02Fztz11', '18S02Fztz12', '18S02Fztz13', '18S02Fztz14', '18S02Fztz15', '18S02Fc0',
          '18S02Fc1', '18S02Fc2', '18S02Fc3', '18S02Fc4', '18S02Fc5', '18S02Fc6', '18S02Fc7', '18S02Fc8', '18S02Fc9',
          '18S02Fc10', '18S02Fc11', '18S02Fc12', '18S02Fc13', '18S02Fc14',
          'signal', '17S03Time', '17S03NDA', '17S03SJ', '17S03NP', '17S03Mxsyp', '17S03MZLsyXp',
          '17S03Mxsyn', '17S03MZLsyXn', '17S03Mysyp', '17S03MZLsyYp', '17S03Mysyn',
          '17S03MZLsyYn', '17S03Mzsyp', '17S03MZLsyZp', '17S03Mzsyn', '17S03MZLsyZn',
          '17S03Dypj', '17S03Dypc', '17S03Dxj', '17S03Dxc', '17S03Dyj', '17S03Dyc',
          '17S03Dzj', '17S03Dzc', '17S03TDY1', '17S03TDX', '17S03TDZ', '17S03TDY',
          '17S03Mxtjp', '17S03MZLtjXp', '17S03Mxtjn', '17S03MZLtjXn', '17S03Mytjp',
          '17S03MZLtjYp', '17S03Mytjn', '17S03MZLtjYn', '17S03Mx/ytjp', '17S03Ijjx', '17S03Ijjy'
    , '17S03Ijjz', '17S03MZLsyX', '17S03MZLsyY', '17S03MZLsyZ', '17S03MZLtjX', '17S03MZLtjY'
    , '17S03MZLsyX(1s)', '17S03MZLsyY(1s)', '17S03MZLsyZ(1s)', '17S03MZLtjX(1s)', '17S03MZLtjY(1s)'
          ]
testcommon = row_data[column]
print('testcommon', testcommon.shape)
a = testcommon[44000:]
b = testcommon[:6000]
# testvalue = a.append(testerror.append(b.append(testerror.append(a[:len(a) - 886]))))
testvalue = a[:4000].append(testerror.append(b[:4000].append(testerror.append(a[:5808]))))
print('testvalue', testvalue.shape)
nocheck = testvalue
row_data = pd.read_excel('/root/xulun/316/20180514/data/validdf.xlsx')
column = [ '18S04Ijmb', '18S04Ts', '18S04Ftm', '18S04Fjm',
          '18S04F2k', '18S04F4k', '18S04F8k',
          '18S04UAm', '18S04UBm', '18S04UCm', '18S04UAa',
          '18S04UBa', '18S04UCa', '18S04U2k', '18S04U4k', '18S04U8k',
          '18S04V5p', '18S04V12p', '18S04V12n', '18S04Vsy15p',
          '18S04Vsy15n', '18S04Vc15p', '18S04Vc15n',
          '18S04Itma', '18S04Itmb', '18S04Itmc', '18S04Ijma', '18S04Ijmc',
          #      '18S04Zkz','18S04Ffz',
          '18S04Ncrc', '18S04Tt', '18S04Txt', '18S04Tyt', '18S04Tzt',
          '18S04Txj', '18S04Tyj', '18S04Iwk', '18S04Wxt', '18S04Wyt', '18S04Wzt', '18S04Wxj', '18S04Wyj',
          '18S04Ix', '18S04Iy','18S04Iz',
          '18S04Iy', '18S04Nw', '18S04Nwc', '18S04Ntc', '18S04Tb', '18S04Ny', '18S04N422', '18S04Zkzb0', '18S04Zkzb1',
          '18S04Zkzb2', '18S04Zkzb3', '18S04Zkzb4', '18S04Zkzb5', '18S04Zkzb6',
          '18S04Zkzb7', '18S04Zkzb8', '18S04Zkzb9', '18S04Zkzb10', '18S04Zkzb11_13',
          '18S04Zkzb14', '18S04Ffzb0', '18S04Ffzb1', '18S04Ffzb2', '18S04Ffzb3',
          '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ffzb3', '18S04Ncrcb0',
          '18S04Ncrcb1', '18S04Ncrcb2', '18S04Ncrcb3', '18S04Ncrcb4', '18S04Ncrcb5',
          '18S04Ncrcb6', '18S04Ncrcb7',
          '18S02Fztz', '18S02Fjd', '18S02Nsa2', '18S02Fc', '18S02Fztz0', '18S02Fztz1', '18S02Fztz2',
          '18S02Fztz3', '18S02Fztz4', '18S02Fztz5', '18S02Fztz6', '18S02Fztz7', '18S02Fztz8', '18S02Fztz9',
          '18S02Fztz10', '18S02Fztz11', '18S02Fztz12', '18S02Fztz13', '18S02Fztz14', '18S02Fztz15', '18S02Fc0',
          '18S02Fc1', '18S02Fc2', '18S02Fc3', '18S02Fc4', '18S02Fc5', '18S02Fc6', '18S02Fc7', '18S02Fc8', '18S02Fc9',
          '18S02Fc10', '18S02Fc11', '18S02Fc12', '18S02Fc13', '18S02Fc14',
          'signal', '17S03Time', '17S03NDA', '17S03SJ', '17S03NP', '17S03Mxsyp', '17S03MZLsyXp',
          '17S03Mxsyn', '17S03MZLsyXn', '17S03Mysyp', '17S03MZLsyYp', '17S03Mysyn',
          '17S03MZLsyYn', '17S03Mzsyp', '17S03MZLsyZp', '17S03Mzsyn', '17S03MZLsyZn',
          '17S03Dypj', '17S03Dypc', '17S03Dxj', '17S03Dxc', '17S03Dyj', '17S03Dyc',
          '17S03Dzj', '17S03Dzc', '17S03TDY1', '17S03TDX', '17S03TDZ', '17S03TDY',
          '17S03Mxtjp', '17S03MZLtjXp', '17S03Mxtjn', '17S03MZLtjXn', '17S03Mytjp',
          '17S03MZLtjYp', '17S03Mytjn', '17S03MZLtjYn', '17S03Mx/ytjp', '17S03Ijjx', '17S03Ijjy'
    , '17S03Ijjz', '17S03MZLsyX', '17S03MZLsyY', '17S03MZLsyZ', '17S03MZLtjX', '17S03MZLtjY'
    , '17S03MZLsyX(1s)', '17S03MZLsyY(1s)', '17S03MZLsyZ(1s)', '17S03MZLtjX(1s)', '17S03MZLtjY(1s)'
          ]
validerror = row_data[column]

# validvalue = b.append(validerror.append(a.append(validerror.append(b[:len(b) - 884]))))
validvalue = b[:4000].append(testerror.append(a[:4000].append(testerror.append(b[:5808]))))
print('这是规模', 'values6=values2:', validvalue.shape, 'values4', testvalue.shape, trainvalue.shape)
# 80%故障加10w正常 10%故障+2000正常测试
###df1训练  df2测试

average = trainvalue['18S04Ijmb'].mean()
print("average = ", average)

#############################################################################
new_trainvalue = pd.DataFrame()
new_testvalue = pd.DataFrame()
new_validvalue = pd.DataFrame()
for t in timestep:
    #t = i * T
    print( 't=', t)
    best_weights_filepath = '/root/xulun/316/20180514/data/2best_weights%0.2f.hdf5' % (t)

    # 取整
    # if ((len(trainvalue) % t) != 0):
    #     new_trainvalue = trainvalue[:int(len(trainvalue) / t) * t]
    # else:
    #     new_trainvalue = trainvalue
    # if ((len(testvalue) % t) != 0):
    #     new_testvalue = testvalue[:int(len(testvalue) / t) * t]
    # else:
    #     new_testvalue = testvalue
    # if ((len(validvalue) % t) != 0):
    #     new_valid = validvalue[:int(len(validvalue) / t) * t]
    # else:
    #     new_valid = validvalue

    new_trainvalue = trainvalue
    new_testvalue = testvalue
    new_validvalue = validvalue
    print('shape check')
    print('train shape', new_trainvalue.shape)
    print('test shape', new_testvalue.shape)
    print('valid shape', new_validvalue.shape)
    print(new_trainvalue['18S04Ijmb'].min(), new_trainvalue['18S04Ijmb'].max())
    ymin = new_testvalue['18S04Ijmb'].min()
    ymax = new_testvalue['18S04Ijmb'].max()
    copytestvalue = testvalue
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled1 = scaler.fit_transform(new_trainvalue)
    # reframed = series_to_supervised(scaled, 1, 1)
    new_trainvalue = scaled1
    new_testvalue = scaler.fit_transform(new_testvalue)
    new_valid = scaler.fit_transform(new_validvalue)


    train_data =[]
    test_data = []
    valid_data = []
    for i in range(len(new_trainvalue)-t):
        train_data.append(new_trainvalue[i:i+t,:160])
    train_X, train_y = train_data, new_trainvalue[t:,0]
    for i in range(len(new_testvalue)-t):
        test_data.append(new_testvalue[i:i+t,:160])
    test_X1, test_y1 = test_data, new_testvalue[t:,0]
    for i in range(len(new_valid)-t):
        valid_data.append(new_valid[i:i+t,:160])
    valid_X, valid_y = valid_data, new_valid[t:,0]
    #train_X, train_y = new_trainvalue[:len(new_trainvalue) - t, :160], new_trainvalue[t:, 0]
    print('this is train_x')
    print(len(train_X))
    print('this is train_y')
    print(len(train_y))
    # test_X1, test_y1 = new_testvalue[:len(new_testvalue) - t, :160], new_testvalue[t:, 0]
    # valid_X, valid_y = new_valid[:len(new_valid) - t, :160], new_valid[t:, 0]
    #print('x', test_X1.shape, valid_X.shape, train_X.shape, 'y', test_y1.shape, valid_y.shape, train_y.shape)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X1 = np.array(test_X1)
    test_y1= np.array(test_y1)
    valid_X = np.array(valid_X)
    valid_y = np.array(valid_y)
    print('trainx shape',train_X.shape,train_y.shape)
    # test_X2, test_y2 = values5[68000:68700, 6:], values5[68000 + t : 68700 + t, 0]
    # no_X1,no_y1 = nocheck[:len(nocheck) - t, :],nocheck[t:,0]
    # reshape input to be 3D [samples, timesteps, features]
    #train_X = train_X.reshape((int(train_X.shape[0] / t), t, 160))
    train_X = train_X.reshape(train_X.shape[0], t, 160)
    #train_y = train_y.reshape(train_y.shape[0],train_y.shape[1], 1)
    valid_X = valid_X.reshape(valid_X.shape[0], t, 160)
    #test_y1 =  test_y1.reshape( test_y1.shape[0],  test_y1.shape[1], 1)
    test_X1 = test_X1.reshape(test_X1.shape[0] , t, 160)
    #valid_y = valid_y.reshape(valid_y.shape[0], valid_y.shape[1], 1)
    #train_y = picky(train_y, t)
    #test_y1 = picky(test_y1, t)
    #valid_y = picky(valid_y, t)
    # train_y = train_y.reshape((int(train_y.shape[0] / t), t, 1))
    # valid_y = valid_y.reshape((int(valid_y.shape[0] / t), t, 1))
    # test_y1 = test_y1.reshape((int(test_y1.shape[0] / t), t, 1))
    #print('x', test_X1.shape, valid_X.shape, train_X.shape ,'y',test_y1.shape , valid_y.shape, train_y.shape)


    ###################################################################每个epoch输出一个auc

    class RocAucMetricCallback(keras.callbacks.Callback):
        i=0
        def __init__(self, validation_data):
            self.x_val, self.y_val = validation_data
            self.i = 0


        def on_epoch_begin(self, epoch, logs={}):
            # 添加roc_auc_val属性
            self.starttime = time.time()

            if not ('roc_auc_val' in self.params['metrics']):
                self.params['metrics'].append('roc_auc_val')
            if not ('costtime' in self.params['metrics']):
                self.params['metrics'].append('costtime')
            if not ('accuracy_score_val' in self.params['metrics']):
                self.params['metrics'].append('accuracy_score_val')
            return self.starttime

        def on_epoch_end(self, epoch, logs={}):
            #starttime = self.on_epoch_begin(epoch)
            self.i = self.i +1
            self.nowtime = time.time()
            self.costtime = self.nowtime - self.starttime


            y_pre = model.predict(self.x_val)
            l = y_pre - self.y_val
            logs['accuracy_score_val'] = float('-inf')
            logs['roc_auc_val'] = float('-inf')
            logs['loss'] = float('-inf')

            if (self.validation_data):
                m = []
                bb = []
                tt = []
                '''
                   for k in self.y_val:
                    if ((265.5 - 93) * k + 93) >= 160:
                        bb.append([1])
                    else:
                        bb.append([0])
                m = np.array(bb)
                logs['roc_auc_val'] = roc_auc_score(m, y_pre.flatten())
                '''


                #file.write(str(self.i)+' '+ str(self.costtime) + '\r\n')
                #costtime.to_excel('/root/xulun/second/costtime%0.2f.xlsx' % (t+i/100), index_label=['1'], index=True)
                logs['costtime'] = self.costtime
                m.append(self.costtime)
            print('time:{costtime}'.format(costtime=logs.get('costtime')))


    my = RocAucMetricCallback(validation_data=(valid_X, valid_y))

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='min')
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                    mode='min', period=1)
    callbacks_list = [my,earlyStopping, saveBestModel]
    #####导入模型
    print('导入模型')
    model = Sequential()
    model.add(LSTM(256, input_shape=(t, train_X.shape[2]), return_sequences=False,dropout=0.5))  # 神经元数修改
    #model.add(LSTM(128, input_shape=(t, train_X.shape[2])))
    model.add(Dense(1))

    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adam')
    print('开始训练')
    # fit network
    begintime = time.time()
    #file = open('/root/xulun/316/20180514/data/costtime%0.2f.txt' % (t ), 'w')
    start_time = time.time()
    history = model.fit(train_X, train_y, epochs=1,batch_size=512, validation_data=(valid_X, valid_y), verbose=2,
                        shuffle=False, callbacks=callbacks_list)  # 训练周期和批量
    #file.close()
    end_time = time.time()
    print('训练时间为:',end_time - start_time)
    file = open('/root/xulun/316/20180514/data/2loss%0.2f.txt' % (t ), 'w')
    file.write('loss:' + str(history.history['loss']) + '  ' + 'valloss:' + str(history.history['val_loss']))
    file.close()
    endtime = time.time()
    time1 = []
    time1.append(endtime - begintime)
    for i in model.layers:
        print('模型参数为：',i.name,i.count_params())
    alltime = pd.DataFrame()
    alltime = alltime.append(time1)
    alltime.to_excel('/root/xulun/316/20180514/data/2traintime%0.2f.xlsx' % (t), index_label=['1'], index=True)
    ########################################故障预测
    # print("lstm", lstm)
    print("over")
    #model.load_weights(best_weights_filepath)
    test_start_time = time.time()
    predict_y1 = model.predict(test_X1)
    # predict_y2 = model.predict(no_X1)
    test_end_time = time.time()
    test_time  = test_end_time - test_start_time
    print('测试时间为：',test_time )
    test_time = pd.DataFrame([test_time])
    test_time.to_excel('/root/xulun/316/20180514/data/2testtime%0.2f.xlsx' % (t), index_label=['1'], index=True)
    # print(predict_y1.shape)
    # y =np.array(predict_y1)
    # #y= y.reshape(len(y),t)
    # print(y.shape)
    # y = pd.DataFrame(y)
    # # predict_y2 = pd.DataFrame(predict_y2)
    # #y.columns = ['y','0']
    # y.to_excel('/root/xulun/316/20180514/data/predict_y%0.2f.xlsx' % (t ), index_label=['1'], index=True)
    # # predict_y2.columns = ['y']
    # # predict_y2.to_excel('/root/xulun/predict_y2 %0.2f.xlsx' % (t + i / 100), index_label=['1'], index=True)
    # print("存储成功！")

    c1 = []
    c2 = []
    d1 = []
    d2 = []
    # 103.5x+162
    # 172.5x+93
    np.set_printoptions(threshold=np.inf)
    #test_y1 = np.array(test_y1)
    #predict_y1 = np.array(predict_y1)

    # d2 = (265.5 - 93) * predict_y2 + 93
    #print('this is 故障预测1', predict_y1[0])
    # y = (265.5 - 93) * trainvalue['18S04Ijmb'] + 93
    predict_y1 = np.array(predict_y1)
    #predict_y1 = predict_y1.reshape(len(predict_y1), t)
    print('predict_y1.shape',predict_y1.shape)
    # predict_y1 = pd.DataFrame(predict_y1)
    # predict_y1 = np.array(predict_y1)
    test_y1 = np.array(test_y1)
    #test_y1 = test_y1.reshape(len(test_y1), t)
    test_y1 = pd.DataFrame(test_y1)
    c1 = (ymax - ymin) * test_y1 + ymin
    # c2 = (265.5 - 93) * no_y1 + 93
    # print('this is 故障实际1', test_y1)
    d1 = (ymax - ymin) * predict_y1 + ymin

    train_y = np.array(train_y)
    #train_y = train_y.reshape(len(train_y),t)
    #train_y = pd.DataFrame(train_y)

    # test_y1.to_excel('/root/xulun/316/20180514/data/test_y%0.2f.xlsx' % (t), index_label=['1'], index=True)
    test_y1 = np.array(test_y1)
    # train_y = np.array(train_y)

    error1 = []
    maerror1 = []

    for m in range(len(test_y1) - 1):
        error1.append(test_y1[m] - predict_y1[m])
        maerror1.append(abs(test_y1[m] - predict_y1[m]))
    squaredError1 = []
    for val in error1:
        squaredError1.append(val * val)  # target-prediction之差平方
    rmse = sqrt(sum(squaredError1) / len(squaredError1))
    mae = (sum(maerror1) / len(maerror1))
    print("RMSE1 = ", rmse)  # 均方根误差RMSE
    print("MAE1=", mae)

    # file = open('/root/xulun/316/20180514/data/mae and rmse%0.2f.txt' % (t), 'w')
    # file.write('rmse:' +str(rmse) + ' ' +'mae:'+str(mae))
    # file.close()
    # # print("第 %f 轮已经结束" % (t + i 100))
    # #########################################正常预测
    # # print('emergence data predict end')
    # plt.plot(train_y[:], label='train label')
    # plt.legend()
    # plt.savefig('/root/xulun/316/20180514/data/data%0.2f.png' % (t ))
    # plt.show()
    # plt.plot(d1[:], label='predict')
    # plt.plot(c1[:], label='actual')
    # plt.legend(loc="lower left")
    # plt.legend()
    # plt.savefig('/root/xulun/316/20180514/data/predictbig%0.2f.png' % (t ))
    # plt.show()
    # plt.plot(predict_y1, label='predict')
    # plt.plot(test_y1, label='label')
    # plt.legend(loc="lower left")
    # plt.legend()
    # plt.savefig('/root/xulun/316/20180514/data/predictsmall%0.2f.png' % (t ))
    # plt.show()
    #
    # plt.plot(history.history['loss'], '--m', label='train', )
    # plt.plot(history.history['val_loss'], ':b', label='test')
    # plt.legend()
    # plt.savefig('/root/xulun/316/20180514/data/loss%0.2f.png' % (t ))
    # plt.show()
    # print('all data predict end')
    '''
           b2 = []
    for k in c1:
        if k >= 160:
            b2.append([1])
        else:
            b2.append([0])
    t1 = np.array(b2)
    auc1 = roc_auc_score(t1, predict_y1)
    print("使用这个模型预测的准确率为", auc1)
    fpr, tpr, _ = metrics.roc_curve(t1, predict_y1)
    plt.subplots(figsize=(7, 5.5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.legend()
    plt.savefig('/root/xulun/roc %0.2f.png' % (t + i / 100))
    plt.show()
    auc1 = roc_auc_score(t1, predict_y1)
    print("使用这个模型预测的准确率为", auc1)

    '''
    print("第 %f 轮已经结束" % (t ))

###############roc曲线

############0
###全部特征，只使用dropout和l2优化解决过拟合，之前代码不全了
############8

