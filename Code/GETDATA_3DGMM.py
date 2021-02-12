import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import multivariate_normal
import cv2

N = 99  # Number of training samples
K = 3  # Number of groups
D = 3  # Number of dimensions
img_R = cv2.imread('TrainingImages/Red.png')
img_G = cv2.imread('TrainingImages/Green.png')
img_Y = cv2.imread('TrainingImages/Yellow.png')
img_W1 = cv2.imread('TrainingImages/Water1.png')
img_W2 = cv2.imread('TrainingImages/Water2.png')
img_W3 = cv2.imread('TrainingImages/Water3.png')
img_W4 = cv2.imread('TrainingImages/Water3_1.png')
x1_ = np.nonzero(img_R)
x2_ = np.nonzero(img_G)
x3_ = np.nonzero(img_Y)
x4_ = np.nonzero(img_W1)
x5_ = np.nonzero(img_W2)
x6_ = np.nonzero(img_W3)
x7_ = np.nonzero(img_W4)
x1 = np.vstack([img_R[x1_[0], x1_[1], 0], img_R[x1_[0], x1_[1], 1], img_R[x1_[0], x1_[1], 2]]).T
x2 = np.vstack([img_G[x2_[0], x2_[1], 0], img_G[x2_[0], x2_[1], 1], img_G[x2_[0], x2_[1], 2]]).T
x3 = np.vstack([img_Y[x3_[0], x3_[1], 0], img_Y[x3_[0], x3_[1], 1], img_Y[x3_[0], x3_[1], 2]]).T
x4 = np.vstack([img_W1[x4_[0], x4_[1], 0], img_W1[x4_[0], x4_[1], 1], img_W1[x4_[0], x4_[1], 2]]).T
x5 = np.vstack([img_W2[x5_[0], x5_[1], 0], img_W2[x5_[0], x5_[1], 1], img_W2[x5_[0], x5_[1], 2]]).T
x6 = np.vstack([img_W3[x6_[0], x6_[1], 0], img_W3[x6_[0], x6_[1], 1], img_W3[x6_[0], x6_[1], 2]]).T
x7 = np.vstack([img_W4[x7_[0], x7_[1], 0], img_W4[x7_[0], x7_[1], 1], img_W3[x7_[0], x7_[1], 2]]).T
#Changing x[1...7] we can get the Mean and Co-variance for the Buoys and Water Segmentation.
#Generated Mean and co-variance acts as a data for 3D-Segmentation of Buoy using GMM.
x = np.vstack([x6])
N = x.shape[0]
Cov_ = [(np.random.randint(100, 200) * np.eye(D)) for i in range(K)]
regCov = 1e-6 * np.eye(D)
#2
Mean_ = np.asarray([[100,230,225],
                    [140,160,225],
                    [150,180,150]])
Gauss_ = [multivariate_normal(x, y) for x, y in zip(Mean_, Cov_)]
Pi_ = np.asarray([1 / K for i in range(K)])
alpha = np.zeros((N, K))
pdf = np.zeros((N, K))
#1
# Mean_ = np.asarray([[2500,240,240],
#                     [150,250,75],
#                     [150,250,75]])
#3
# Mean_ = np.asarray([[175, 200, 200],
#                      [100, 180, 140],
#                      [250, 240, 240]])


def projection(x,x1,x2,x3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, m, x_ in zip(['r', 'g', 'y'], ['o', '^', '*'], [x1, x2, x3]):
        plt.plot(x_[:, 0], x_[:, 1], x_[:, 2], c=c, marker=m, alpha=0.1)
    ax.set_xlabel('xAxis')
    ax.set_ylabel('yAxis')
    ax.set_zlabel('zAxis')
    plt.show()


def Generate_Data(Gauss_,Pi_,Mean_):
    i = 300
    for itera in range(i):
        for i, gauss in enumerate(Gauss_):
            pdf[:, i] = gauss.pdf(x)
        pixPdf = np.dot(pdf, np.diag(Pi_))
        pixPdfSum = np.sum(pixPdf, axis=-1)
        alpha = np.divide(pixPdf.T,
                          pixPdfSum + 1e-8).T

        alphaSum = np.sum(alpha.T, axis=-1)
        mean = np.divide((np.dot(alpha.T, x)).T, alphaSum + 1e-8).T
        cov = []
        for i in range(K):
            xSubMu = x - Mean_[i, :]
            alphaTemp = np.multiply(alpha[:, i], np.ones((N, D)).T)
            cov.append(np.divide(np.dot(xSubMu.T, np.multiply(alphaTemp.T, xSubMu)), np.sum(alpha[:, i]) + 1e-8))
            cov[-1] += regCov
        pi = alphaSum / N
        Mean_ = copy.deepcopy(mean)
        Cov_ = copy.deepcopy(cov)
        Pi_ = copy.deepcopy(pi)
        Gauss_ = [multivariate_normal(mu_, cov_) for mu_, cov_ in zip(Mean_, Cov_)]

    return mean,cov

projection(x,x1,x2,x3)
mean,cov = Generate_Data(Gauss_,Pi_,Mean_)
print("Mean",mean)
print("Co-variance",cov)