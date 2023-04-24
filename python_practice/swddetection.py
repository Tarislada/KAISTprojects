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
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf
from imblearn.under_sampling import InstanceHardnessThreshold as IHT

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
dir = 'C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/바탕 화면/EEG_rawdata_YW/alleegdata/'
answersheet = sio.loadmat('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/문서/카카오톡 받은 파일/ywt.mat')

base = '22721005'
data1 = '22721007'
data2 = '22721009'
data3 = '22727006'
data4 = '22720014'
control2 = '22721003'
control3 = '22727008'
control4 = '22727002'

answermat1 = 'CaV13'
answermat2 = 'CaV14'
answermat3 = 'CaV15'
answermat4 = 'CaV12'
controlanswermat2 = 'Con2'
controlanswermat3 = 'Con3'
controlanswermat4 = 'Con4'

data1abf = pyabf.ABF(dir+data1+'.abf')
data2abf = pyabf.ABF(dir+data2+'.abf')
data3abf = pyabf.ABF(dir+data3+'.abf')
data4abf = pyabf.ABF(dir+data4+'.abf')
data1ch1signal = data1abf.data[0,:]
data1ch2signal = data1abf.data[1,:]
# data2ch1signal = data2abf.data[0,:]
# data2ch2signal = data2abf.data[1,:]
# data3ch1signal = data3abf.data[0,:]
# data3ch2signal = data3abf.data[1,:]
# data4ch1signal = data4abf.data[0,:]
# data4ch2signal = data4abf.data[1,:]
# base1signal = abf2.data[0,:]
# base2signal = abf2.data[1,:]

con2abf = pyabf.ABF(dir+control2+'.abf')
con3abf = pyabf.ABF(dir+control3+'.abf')
con4abf = pyabf.ABF(dir+control4+'.abf')
con2ch1signal = con2abf.data[0,:]
con2ch2signal = con2abf.data[1,:]
con3ch1signal = con3abf.data[0,:]

def run_fcwt(inputdata):

    # Define a second-order butterworth lowpass filter
    def butter_lowpass(data,cutoff,fs):
        normal_cutoff = cutoff/2
        b,a = butter(2,normal_cutoff,btype='low',analog=False,fs=fs)
        y = filtfilt(b,a,data)
        return y
# Scale raw eeg signals
    scaled = (inputdata-np.mean(inputdata))/np.std(inputdata)

# Apply lowpass filter
    lwpssed = butter_lowpass(scaled,40,fs)
    lwpssed = lwpssed.astype(np.float32)

# Set parameters for fcwt
    f0 = 1 #lowest frequency
    f1 = 20 #highest frequency
    fn = 140 #number of frequencies
    morl = fcwt.Morlet(2.0)
    linscales = fcwt.Scales(morl, fcwt.FCWT_LINFREQS, fs, f0, f1, fn)

    nthreads = 8
    use_optimization_plan = False
    use_normalization = True
    fcwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)

# Initialize output array
    output = np.zeros((fn,lwpssed.size), dtype=np.complex64)
    fcwt_obj.cwt(lwpssed, linscales, output)
    return output

def createlabel(datasize,answers,fs):
    """

    :param data: data to be labeled ex) data1ch1signal.size
    :param answers: ex) answersheet[answermat1]
    :return: label vector
    """
    bindow = 15
    tmplabel = np.zeros(datasize)
    for i in range(answers.shape[0]):
        tmplabel[int(answers[i, 0] * fs):int(answers[i, 1] * fs)] = 1

    label = scipy.ndimage.uniform_filter1d(tmplabel,bindow)
    label = label[1::bindow]
    label = (np.rint(label)).astype(int)
    return label, tmplabel

# Create label vector

# Shape dataset for xgboost
def clean_data(inputdata1,inputdata2):
    """
    binning & standardization & pca
    :param inputdata1:
    :param inputdata2:
    :return:
    """
    bindow = 15
    # output1 = scipy.ndimage.uniform_filter(inputdata1, [bindow, 1])
    # output2 = scipy.ndimage.uniform_filter(inputdata2, [bindow, 1])
    # output1 = output1[:,1::bindow]
    # output2 = output2[:,1::bindow]
    scaler1 = sklearn.preprocessing.StandardScaler()
    cleaned_data = np.hstack((scaler1.fit_transform(inputdata1.transpose().real),
                       scaler1.fit_transform((inputdata2.transpose().real))))
    smoothed_data = scipy.ndimage.uniform_filter1d(cleaned_data,bindow)
    output = smoothed_data[:,1::bindow]
    # smoothed_data = scipy.ndimage.gaussian_filter1d(cleaned_data,)
    # pca = decomposition.PCA(n_components=.95)
    # cleaned_data = pca.fit_transform(data1)
    # return output
    return smoothed_data
# TODO: Data normalization
def stack_data(inputdata1,inputdata2):
    stacked = np.hstack((inputdata1,inputdata2))
    return stacked

# data = np.abs(data1[:,600*2000:1200*2000])
def run_xgboost(data,label):
    data = data[600 * 2000:1200 * 2000, :]
    label = label[600 * 2000:1200 * 2000]

    # Set model params and evaluate parameters via 5fold cross validation
    # data_mat = xgb.DMatrix(data=data.transpose(),label=label)
    # data_mat = xgb.DMatrix(data=data,label=label)
    params = {"objective": "binary:logistic", 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5,
              'alpha': 10, "eval_metric": "auc","scale_pos_weight": 10}
    # xgb_cv = xgb.cv(dtrain=data_mat,params=params,nfold=5,num_boost_round=100,early_stopping_rounds=10,metrics="auc",as_pandas=True,seed=123)
    # xgb_class = xgb.XGBClassifier(**params)
    # xgb_cv.head()

    # Train model
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(np.real(data), np.real(label.transpose()), test_size=0.2,
                                                                                random_state=0)
    # xgb_class.fit(X_train, y_train)
    # y_pred = xgb_class.predict(X_test)
    # xgb_class.fit(data.transpose(), label)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    clf = xgb.XGBClassifier(**params)
    fitted = clf.fit(X=X_train, y=y_train)
    iht = IHT(estimator=fitted,random_state=42)
    X_train_res,y_train_res = iht.fit_resample(X_train,y_train.astype(int))

    model = xgb.train(dtrain=dtrain, params=params, num_boost_round=100, verbose_eval=10, evals=watchlist)

    return model


def run_cnnclassifier(data,label):
    bindow = 15
    data = data[int(600 * 2000/bindow):int(1200 * 2000/bindow), :]
    label = label[ int(600 * 2000/bindow):int(1200 * 2000/bindow)]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label.transpose(), test_size=0.2,
                                                                                random_state=0)
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryCrossentropy()])
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    return test_loss, test_acc, model

# TODO: training only on one sample: 22721007. Need to see if it applies to other samples - it doesn't
# checktreated = '22720012'
# checkanswermat = 'CaV11'
#
# checkabf = pyabf.ABF(dir+checktreated+'.abf')
# checktmpdata = checkabf.data
# check1signal = checktmpdata[0,:]
# check2signal = checktmpdata[1,:]

checkcontrol = con4abf
checkcon1sig = checkcontrol.data[0,:]
checkcon2sig = checkcontrol.data[1,:]

#dval = xgb.DMatrix(checkdata,label=checklabel)

# y_pred = model.predict(dval)
#TODO: NOT WORKING WHY???? REVERT TO V1 POSSIBLE CAUSE: fit vs train? basesignal based normalization? dimension reduction? Binning?
#TODO: Moving average and
#TODO: Dimension reduction worked, but insufficient. convert to original feature space? try other demension reductions and binning
#TODO: Try not pca, but raw?


# cwtsignal1 = run_fcwt(data1ch1signal)
# cwtsignal2 = run_fcwt(data1ch2signal)
# label = createlabel(data1ch1signal.size,answersheet[answermat1],fs)[0]
# cleaneddata = clean_data(cwtsignal1,cwtsignal2)
# pca = decomposition.PCA(n_components=.95)
# pcaeddata = pca.fit_transform(cleaneddata)
# loss,acc,cnnmodel =run_cnnclassifier(pcaeddata,label)
#
# testsig1 = run_fcwt(check1signal)
# testsig2 = run_fcwt(check2signal)
# testlabel = createlabel(check1signal.size,answersheet[checkanswermat],fs)[0]
# cleanedtest = clean_data(testsig1,testsig2)
# pcaedtest = pca.transform(cleanedtest)
# checkdata = np.abs(pcaedtest[600*2000:1200*2000,:])
# checklabel = testlabel = testlabel[600*2000:1200*2000]

con2cwtsig1 = run_fcwt(con2ch1signal)
con3cwtsig1 = run_fcwt(con3ch1signal)
_,con2label = createlabel(con2ch1signal.size,answersheet[controlanswermat2],fs)
_,con3label = createlabel(con3ch1signal.size,answersheet[controlanswermat3],fs)
# stacked = np.vstack((np.real(cwtsig1),np.real(cwtsig2))).transpose()
con2stacked = np.vstack((np.real(con2cwtsig1),np.imag(con2cwtsig1))).transpose()
con3stacked = np.vstack((np.real(con3cwtsig1),np.imag(con3cwtsig1))).transpose()
# cleaneddata = clean_data(cwtsig1,cwtsig2)
# pca = decomposition.PCA(n_components=.95)
# pcadata = pca.fit_transform(cleaneddata)
# xgboostmodel = run_xgboost(pcadata,Zabel)
xgb_nopca = run_xgboost(con2stacked,con2label)

testsig1 = run_fcwt(checkcon1sig)
testsig2 = run_fcwt(checkcon2sig)
_,testlabel = createlabel(checkcon1sig.size,answersheet[controlanswermat4],fs)
# teststacked = np.vstack((np.real(testsig1),np.real(testsig2))).transpose()
teststacked = np.vstack((np.real(testsig1),np.imag(testsig1),np.real(testsig2),np.imag(testsig2))).transpose()
# cleanedtest = clean_data(testsig1,testsig2)
# pcatest = pca.transform(cleanedtest)
# # checkdata = (pcatest[600*2000:1200*2000,:])
checkdata = teststacked[600*2000:1200*2000,:]
checklabel = testlabel[600*2000:1200*2000]


# ls,acc,cnnmodel = run_cnnclassifier(pcadata,label)
y_pred = xgb_nopca.predict(xgb.DMatrix(checkdata))
fpr,tpr,thres = sklearn.metrics.roc_curve(checklabel,y_pred)
aucscore = sklearn.metrics.auc(fpr,tpr)
y_pred = cnnmodel.predict()

# Todo: Change to Normal, not CAV / smoothing rather than binning?
# NOpca>pca
# Increasing the number of features V
# Clean not only cwt-ed eeg, clean eeg itself V
# imaginary part information? include by treating it as a feature of a sample (vstack it under same label) V
# imbalance when doing it w classifiers - include samples from other dataset + imblearn IHT + xgboost scales
# average 2 channels
# True tuning w/ hyperparameters https://stackoverflow.com/questions/68766331/how-to-apply-predict-to-xgboost-cross-validation