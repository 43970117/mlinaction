import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    with open(fileName) as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        dataArr=[]
        for line in stringArr:
            dataArr.append(list(map(float, line)))
    return np.array(dataArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals,eigVects = np.linalg.eigh(covMat)
    eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = np.matmul(meanRemoved, redEigVects)#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


dataMat = loadDataSet('testSet.txt')
lowDMat, reconMat = pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0], dataMat[:,1], marker='^', s=90)
ax.scatter(reconMat[:,0], reconMat[:,1], marker='o', s=50, c='red')
plt.show()