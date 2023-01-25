import fcwt
import numpy as np
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import butter,filtfilt
import scipy.io as sio
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# Set basic parameters

fs = 2000
# N = 2000
# deltat = N*1/12
# tau = N*6
# offth = 4.5
# onth = 1
# method = 'gaussian'
# pvr = 6
# npeak = 1.5

# Load files

# Load files
dir = 'C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/바탕 화면/EEG_rawdata_YW/alleegdata/'
answersheet = sio.loadmat('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/문서/카카오톡 받은 파일/ywt.mat')

base = '22721005'
treated = '22721007'
answermat = 'CaV13'

abf = pyabf.ABF(dir+treated+'.abf')
tmpdata = abf.data
ch1signal = tmpdata[0,:]
ch2signal = tmpdata[1,:]

# abf2 = pyabf.ABF(dir+base+'.abf')
# base1signal = abf2.data[0,:]
# base2signal = abf2.data[1,:]

# Define a second-order butterworth lowpass filter
def butter_lowpass(data,cutoff,fs):
    normal_cutoff = cutoff/2
    b,a = butter(2,normal_cutoff,btype='low',analog=False,fs=fs)
    y = filtfilt(b,a,data)
    return y

# Apply lowpass filter
ch1lwpssed = butter_lowpass(ch1signal,40,fs)
ch2lwpssed = butter_lowpass(ch2signal,40,fs)
ch1lwpssed = ch1lwpssed.astype(np.float32)
ch2lwpssed = ch2lwpssed.astype(np.float32)

# base1lwpssed = butter_lowpass(base1signal,40,fs)
# base2lwpssed = butter_lowpass(base2signal,40,fs)
# base1lwpssed = base1lwpssed.astype(np.float32)
# base2lwpssed = base2lwpssed.astype(np.float32)

# Set parameters for fcwt
f0 = 2 #lowest frequency
f1 = 20 #highest frequency
fn = 35 #number of frequencies
morl = fcwt.Morlet(2.0)
linscales = fcwt.Scales(morl, fcwt.FCWT_LINFREQS, fs, f0, f1, fn)

nthreads = 8
use_optimization_plan = False
use_normalization = True
fcwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)

# Initialize output array
ch1output = np.zeros((fn,tmpdata.size), dtype=np.complex64)
ch2output = np.zeros((fn,tmpdata.size), dtype=np.complex64)
# base1output = np.zeros((fn,tmpdata.size), dtype=np.complex64)
# base2output = np.zeros((fn,tmpdata.size), dtype=np.complex64)

# Calculate fcwt
fcwt_obj.cwt(ch1lwpssed, linscales, ch1output)
fcwt_obj.cwt(ch2lwpssed, linscales, ch2output)
# fcwt_obj.cwt(base1lwpssed, linscales, ch1output)
# fcwt_obj.cwt(base2lwpssed, linscales, ch1output)

# Create label vector
tmplabel = np.zeros((1,ch1lwpssed.size))
for i in range(answersheet[answermat].shape[0]):
    tmplabel[0,int(answersheet[answermat][i,0]*fs):int(answersheet[answermat][i,1]*fs)]=1

# Shape dataset for xgboost
datastack = np.vstack((ch1output,ch2output))
data = np.abs(datastack[:,600*2000:1200*2000])
label = tmplabel[:,600*2000:1200*2000]

# Set model params and evaluate parameters via 5fold cross validation
data_mat = xgb.DMatrix(data=data.transpose(),label=label)
params = {"objective":"binary:logistic",'colsample_bytree':0.3,'learning_rate':0.1,'max_depth':5,'alpha':10}
xgb_cv = xgb.cv(dtrain=data_mat,params=params,nfold=5,num_boost_round=50,early_stopping_rounds=10,metrics="auc",as_pandas=True,seed=123)
xgb_class = xgb.XGBClassifier(**params)
# xgb_cv.head()

# Train model
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data,label,test_size=0.2, random_state=0)
# xgb_class.fit(X_train, y_train)
# y_pred = xgb_class.predict(X_test)
xgb_class.fit(data.transpose(), label)

# TODO: training only on one sample: 22721007. Need to see if it applies to other samples - it doesn't
# Todo: Solutions - stack more data? adjust parameters?
checktreated = '22720012'
checkanswermat = 'CaV11'

checkabf = pyabf.ABF(dir+checktreated+'.abf')
checktmpdata = checkabf.data
check1signal = checktmpdata[0,:]
check2signal = checktmpdata[1,:]

check1lwpssed = butter_lowpass(check1signal,40,fs)
check2lwpssed = butter_lowpass(check2signal,40,fs)
check1lwpssed = ch1lwpssed.astype(np.float32)
check2lwpssed = ch2lwpssed.astype(np.float32)

check1output = np.zeros((fn,tmpdata.size), dtype=np.complex64)
check2output = np.zeros((fn,tmpdata.size), dtype=np.complex64)

fcwt_obj.cwt(check1lwpssed, linscales, check1output)
fcwt_obj.cwt(check2lwpssed, linscales, check2output)
# fcwt_obj.cwt(base1lwpssed, linscales, ch1output)
# fcwt_obj.cwt(base2lwpssed, linscales, ch1output)

# Create label vector
checktmplabel = np.zeros((1,check1lwpssed.size))
for i in range(answersheet[checkanswermat].shape[0]):
    checktmplabel[0,int(answersheet[checkanswermat][i,0]*fs):int(answersheet[checkanswermat][i,1]*fs)] = 1

# Shape dataset for xgboost
checkdatastack = np.vstack((check1output,check2output))
checkdata = np.abs(checkdatastack[:,600*2000:1200*2000])
checklabel = checktmplabel[:,600*2000:1200*2000]

y_pred = xgb_class.predict(checkdata.transpose())
# checklabel vs y_pred/

