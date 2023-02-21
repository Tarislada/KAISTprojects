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
from sklearn import decomposition
import scipy

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
data1 = '22721007'
data2 = '22721009'
data3 = '22727006'
data4 = '22720014'
answermat1 = 'CaV13'
answermat2 = 'CaV14'
answermat3 = 'CaV15'
answermat4 = 'CaV12'

data1abf = pyabf.ABF(dir+data1+'.abf')
data2abf = pyabf.ABF(dir+data2+'.abf')
data3abf = pyabf.ABF(dir+data3+'.abf')
data4abf = pyabf.ABF(dir+data4+'.abf')
data1ch1signal = data1abf.data[0,:]
data1ch2signal = data1abf.data[1,:]
data2ch1signal = data2abf.data[0,:]
data2ch2signal = data2abf.data[1,:]
data3ch1signal = data3abf.data[0,:]
data3ch2signal = data3abf.data[1,:]
data4ch1signal = data4abf.data[0,:]
data4ch2signal = data4abf.data[1,:]

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
data1ch1lwpssed = butter_lowpass(data1ch1signal,40,fs)
data1ch2lwpssed = butter_lowpass(data1ch2signal,40,fs)
data1ch1lwpssed = data1ch1lwpssed.astype(np.float32)
data1ch2lwpssed = data1ch2lwpssed.astype(np.float32)

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
data1ch1output = np.zeros((fn,data1ch1signal.size), dtype=np.complex64)
data1ch2output = np.zeros((fn,data1ch2signal.size), dtype=np.complex64)
data2ch1output = np.zeros((fn,data2ch1signal.size), dtype=np.complex64)
data2ch2output = np.zeros((fn,data2ch2signal.size), dtype=np.complex64)
data3ch1output = np.zeros((fn,data3ch1signal.size), dtype=np.complex64)
data3ch2output = np.zeros((fn,data3ch2signal.size), dtype=np.complex64)
data4ch1output = np.zeros((fn,data4ch1signal.size), dtype=np.complex64)
data4ch2output = np.zeros((fn,data4ch2signal.size), dtype=np.complex64)
# base1output = np.zeros((fn,tmpdata.size), dtype=np.complex64)
# base2output = np.zeros((fn,tmpdata.size), dtype=np.complex64)

# Calculate fcwt
# fcwt_obj.cwt(ch1lwpssed, linscales, ch1output)
# fcwt_obj.cwt(ch2lwpssed, linscales, ch2output)
fcwt_obj.cwt(data1ch1lwpssed, linscales, data1ch1output)
fcwt_obj.cwt(data1ch2lwpssed, linscales, data1ch2output)
# fcwt_obj.cwt(data2ch1signal, linscales, data2ch1output)
# fcwt_obj.cwt(data2ch2signal, linscales, data2ch2output)
# fcwt_obj.cwt(data3ch1signal, linscales, data3ch1output)
# fcwt_obj.cwt(data3ch2signal, linscales, data3ch2output)
# fcwt_obj.cwt(data4ch1signal, linscales, data4ch1output)
# fcwt_obj.cwt(data4ch2signal, linscales, data4ch2output)

# fcwt_obj.cwt(base1lwpssed, linscales, ch1output)
# fcwt_obj.cwt(base2lwpssed, linscales, ch1output)

# Create label vector
data1tmplabel = np.zeros((1,data1ch1signal.size))
# data2tmplabel = np.zeros((1,data2ch1signal.size))
# data3tmplabel = np.zeros((1,data3ch1signal.size))
# data4tmplabel = np.zeros((1,data4ch1signal.size))
for i in range(answersheet[answermat1].shape[0]):
    data1tmplabel[0,int(answersheet[answermat1][i,0]*fs):int(answersheet[answermat1][i,1]*fs)]=1
# for i in range(answersheet[answermat2].shape[0]):
#     data2tmplabel[0,int(answersheet[answermat2][i,0]*fs):int(answersheet[answermat2][i,1]*fs)]=1
# for i in range(answersheet[answermat3].shape[0]):
#     data3tmplabel[0, int(answersheet[answermat3][i, 0] * fs):int(answersheet[answermat3][i, 1] * fs)] = 1
# for i in range(answersheet[answermat4].shape[0]):
#     data4tmplabel[0, int(answersheet[answermat4][i, 0] * fs):int(answersheet[answermat4][i, 1] * fs)] = 1

# Shape dataset for xgboost
# data1 = np.hstack((sklearn.preprocessing.normalize(np.abs(data1ch1output.transpose())),sklearn.preprocessing.normalize(np.abs(data1ch2output.transpose()))))
bindow = 15
data1ch1outputt = scipy.ndimage.uniform_filter(data1ch1output,[bindow,1])
data1ch2outputt = scipy.ndimage.uniform_filter(data1ch2output, [bindow, 1])

scaler1 = sklearn.preprocessing.StandardScaler()

data1 = np.hstack((scaler1.fit_transform(data1ch1outputt.transpose().real),scaler1.fit_transform((data1ch2outputt.transpose().real))))
# data1 = np.hstack((sklearn.preprocessing.normalize(np.abs(data1ch1output.transpose())),sklearn.preprocessing.normalize(np.abs(data1ch2output.transpose()))))
pca = decomposition.PCA(n_components=.95)
data1 = pca.fit_transform(data1)
# data1 = np.hstack((sklearn.preprocessing.normalize(data1ch1output.transpose()),sklearn.preprocessing.normalize(data1ch2output.transpose())))
# data2 = np.hstack((sklearn.preprocessing.normalize(data2ch1output.transpose()),sklearn.preprocessing.normalize(data2ch2output.transpose())))
# data3 = np.hstack((sklearn.preprocessing.normalize(data3ch1output.transpose()),sklearn.preprocessing.normalize(data3ch2output.transpose())))
# data4 = np.hstack((sklearn.preprocessing.normalize(data4ch1output.transpose()),sklearn.preprocessing.normalize(data4ch2output.transpose())))
# data1stack = np.vstack((data1ch1output,data1ch2output))
# data2stack = np.vstack((data2ch1output,data2ch2output))
# data3stack = np.vstack((data3ch1output,data3ch2output))
# data4stack = np.vstack((data4ch1output,data4ch2output))
# TODO: Data normalization
# data = np.hstack((np.abs(data1[:,600*2000:1200*2000]),np.abs(data2[:,600*2000:1200*2000]),np.abs(data3[:,600*2000:1200*2000]),np.abs(data4[:,600*2000:1200*2000])))
# label = np.hstack((data1tmplabel[:,600*2000:1200*2000],data2tmplabel[:,600*2000:1200*2000],data3tmplabel[:,600*2000:1200*2000],data4tmplabel[:,600*2000:1200*2000]))

# data = np.abs(data1[:,600*2000:1200*2000])
data = data1[600*2000:1200*2000,:]
label = data1tmplabel[:,600*2000:1200*2000]

# Set model params and evaluate parameters via 5fold cross validation
# data_mat = xgb.DMatrix(data=data.transpose(),label=label)
# data_mat = xgb.DMatrix(data=data,label=label)
params = {"objective":"binary:logistic",'colsample_bytree':0.3,'learning_rate':0.1,'max_depth':5,'alpha':10,"eval_metric":"auc"}
# xgb_cv = xgb.cv(dtrain=data_mat,params=params,nfold=5,num_boost_round=100,early_stopping_rounds=10,metrics="auc",as_pandas=True,seed=123)
# xgb_class = xgb.XGBClassifier(**params)
# xgb_cv.head()

# Train model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data,label.transpose(),test_size=0.2, random_state=0)
# xgb_class.fit(X_train, y_train)
# y_pred = xgb_class.predict(X_test)
# xgb_class.fit(data.transpose(), label)
dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test,y_test)
watchlist = [(dtest, "eval"), (dtrain, "train")]
model = xgb.train(params=params,dtrain=dtrain, num_boost_round=50,verbose_eval=10,evals=watchlist)


# TODO: training only on one sample: 22721007. Need to see if it applies to other samples - it doesn't
checktreated = '22720012'
checkanswermat = 'CaV11'

checkabf = pyabf.ABF(dir+checktreated+'.abf')
checktmpdata = checkabf.data
check1signal = checktmpdata[0,:]
check2signal = checktmpdata[1,:]

check1lwpssed = butter_lowpass(check1signal,40,fs)
check2lwpssed = butter_lowpass(check2signal,40,fs)
check1lwpssed = check1lwpssed.astype(np.float32)
check2lwpssed = check2lwpssed.astype(np.float32)

check1output = np.zeros((fn,checktmpdata.size), dtype=np.complex64)
check2output = np.zeros((fn,checktmpdata.size), dtype=np.complex64)

fcwt_obj.cwt(check1signal, linscales, check1output)
fcwt_obj.cwt(check2signal, linscales, check2output)
# fcwt_obj.cwt(base1lwpssed, linscales, ch1output)
# fcwt_obj.cwt(base2lwpssed, linscales, ch1output)

# Create label vector
checktmplabel = np.zeros((1,check1lwpssed.size))
for i in range(answersheet[checkanswermat].shape[0]):
    checktmplabel[0,int(answersheet[checkanswermat][i,0]*fs):int(answersheet[checkanswermat][i,1]*fs)] = 1

# Shape dataset for xgboost
# checkdatastack = np.vstack((check1output,check2output))
# checkdata1 = np.hstack((sklearn.preprocessing.normalize(np.abs(check1output.transpose())),sklearn.preprocessing.normalize(np.abs(check2output.transpose()))))
# checkdata1 = np.hstack((np.abs(check1output.transpose()),np.abs(check2output.transpose())))

check1outputt = scipy.ndimage.uniform_filter(check1output,[bindow,1])
check2outputt = scipy.ndimage.uniform_filter(check2output, [bindow, 1])

scaler2 = sklearn.preprocessing.StandardScaler()
checkdata1 = np.hstack((scaler2.fit_transform(check1outputt.transpose().real),scaler1.fit_transform((check2outputt.transpose().real))))
valpca = decomposition.PCA(n_components=data1.shape[1])
pcaedval = valpca.fit_transform(checkdata1)
checkdata = np.abs(pcaedval[600*2000:1200*2000,:])
# checkdata = np.abs(checkdatastack[:,600*2000:1200*2000])
checklabel = checktmplabel[:,600*2000:1200*2000]
dval = xgb.DMatrix(checkdata,label=checklabel)
y_pred = model.predict(checkdata.transpose())
# y_pred = model.predict(dval)
#TODO: NOT WORKING WHY???? REVERT TO V1 POSSIBLE CAUSE: fit vs train? basesignal based normalization? dimension reduction? Binning?
#TODO: Moving average and
#TODO: Dimension reduction worked, but insufficient. convert to original feature space? try other demension reductions and binning