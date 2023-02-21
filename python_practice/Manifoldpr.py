import scipy
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
import umap.umap_ as umap

def signal_time(importnum):

    # import matrix - sample(time) x feature(neuron)
    allvars = scipy.io.loadmat(
        'C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/endo/datafiles/tmpvars/1007.mat')
    dnsigcell = allvars['dnsigcell']
    dnarray = dnsigcell[0, importnum]

    # Turn timecell into labelvector
    timecell = allvars['timecellcell']
    timearray = timecell[0, importnum]
    rawlabelvec = np.zeros(dnarray.shape[0])
    rawlabelvec[:timearray[0][0][0][0] - 1] = 1
    for ii in range(1, timearray.shape[0]):
        for iii in range(timearray[ii][0].shape[0]):
            rawlabelvec[timearray[ii][0][iii][1] - 1:timearray[ii][0][iii][0] - 1] = ii
                # is this right? MATLAB and python has different indexing systems
    del allvars
    return dnarray,rawlabelvec,timearray

def rundr_pca(dnarray):
    # Clean data - preprocessing with standardization
    scaler1 = preprocessing.StandardScaler()
    testarray_st = scaler1.fit_transform(dnarray)

    # run PCA (or other dimension reduction techniques)
    testpca = decomposition.PCA()
    testpca.fit(testarray_st)
    pca_array = testpca.transform(testarray_st)

    return pca_array

def rundr_pca(dnarray):
    # Clean data - preprocessing with standardization
    scaler1 = preprocessing.StandardScaler()
    testarray_st = scaler1.fit_transform(dnarray)

    # run PCA (or other dimension reduction techniques)
    testpca = decomposition.PCA()
    testpca.fit(testarray_st)
    pca_array = testpca.transform(testarray_st)

    return pca_array

def rundr_isomap(dnarray):
    scaler1 = preprocessing.StandardScaler()
    testarray_st = scaler1.fit_transform(dnarray)

    testiso = manifold.Isomap(n_components=3)
    testiso.fit(testarray_st)
    iso_array = testiso.transform(testarray_st)

    return iso_array

def rundr_hlle(dnarray):
    scaler1 = preprocessing.StandardScaler()
    testarray_st = scaler1.fit_transform(dnarray)

    testhlle = manifold.LocallyLinearEmbedding(n_components=3,method='hessian',n_neighbors=10,eigen_solver='dense')
    testhlle.fit(testarray_st)
    hlle_array = testhlle.transform(testarray_st)

    return hlle_array

def rundr_umap(dn_array):
    scaler1 = preprocessing.StandardScaler()
    testarray_st = scaler1.fit_transform(dnarray)

    testumap = umap.UMAP(n_components=3)
    testumap.fit(testarray_st)
    umap_array = testumap.transform(testarray_st)

    return umap_array

def draw_results(pca_array,timearray,behavnum):
    # select new features or 'neural manifold' space that best describes the matrix
    visualizd = pca_array[:,:3]

    # visualize the time series on new axis
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    for i in range(timearray[behavnum][0].shape[0]):
        behavstrip = range(timearray[behavnum][0][i][1],timearray[behavnum][0][i][0]+1)
        axes.plot3D(visualizd[behavstrip,0],visualizd[behavstrip,1],visualizd[behavstrip,2])

        # 지금은 방향성이 없는 상태. 방향성을 도입하기 위해서는 시작점과 끝 점을 찍어야하고, 각 라인의 색이 달라야한다?
        # PCA only explains 40% of variance; not a good representation. try isomap, HLLE, MDS, LEM, UMAP
        # also, can differ according to learning state
    fig.show()
    return fig, axes

# compare the routes of time series


# run functions
dnarray,rawlabelvec,timearray = signal_time(15)
# dr_array = rundr_pca(dnarray)
dr_array = rundr_isomap(dnarray)
# dr_array = rundr_hlle(dnarray)
# dr_array = rundr_umap(dnarray)
draw_results(dr_array,timearray,2)