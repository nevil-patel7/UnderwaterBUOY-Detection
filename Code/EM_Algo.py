import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import copy
import math


set = int(input("Enter 0 -> 1-D EM \n or \n 1 -> 2-D EM"))

if(set == 0):

    #Generating 3 Random Data
    data1 = np.random.normal(loc=-10, scale=3, size=100)
    data2 = np.random.normal(loc=0, scale=2, size=100)
    data3 = np.random.normal(loc=14, scale=5, size=100)

    # Combine the clusters to get the random datapoints from above
    dataAll = np.stack((data1,data2,data3)).flatten()

    plt.figure(1)
    plt.subplot(311)
    plt.title("Sample Data Set-SINGLE CHANNEL")
    base = np.zeros_like(data1)
    plt.plot(data1,base,"rx")
    plt.subplot(312)
    plt.plot(data2,base,"gx")
    plt.subplot(313)
    plt.plot(data3,base,"bx")

    plt.figure(2)
    plt.title("Sample Data Set-THREE CHANNEL")
    base = np.zeros_like(data1)
    plt.plot(data1,base,"rx")
    plt.plot(data2,base,"gx")
    plt.plot(data3,base,"bx")

    plt.figure(3)
    plt.title("PDF FOR EACH GAUSSIAN")
    x = np.linspace(-30,30, num=300, endpoint=True)
    base = np.zeros_like(data1)
    plt.plot(data1,base,"rx")
    plt.plot(x,norm.pdf(x, -12, 4),"r-",linewidth=1)
    base = np.zeros_like(data2)
    plt.plot(data2,base,"gx")
    plt.plot(x,norm.pdf(x, 0, 3),"g-",linewidth=1)
    base = np.zeros_like(data3)
    plt.plot(data3,base,"bx")
    plt.plot(x,norm.pdf(x, 14, 5),"b-",linewidth=1)

    dataAll.sort()
    # initialization
    m = np.array([1/3,1/3,1/3])
    pi = m / np.sum(m)

    mean = np.array([-5,1,2]).astype(np.float64)
    standDev = np.array([5,3,1]).astype(np.float64)

    plt.figure(4)
    plt.title("INITIAL EM RESULTS")
    base = np.zeros_like(data1)
    plt.plot(data1,base,"gx")
    plt.plot(x,norm.pdf(x, mean[0], standDev[0]),"k-",linewidth=1)
    base = np.zeros_like(data2)
    plt.plot(data2,base,"gx")
    plt.plot(x,norm.pdf(x, mean[1], standDev[1]),"k-",linewidth=1)
    base = np.zeros_like(data3)
    plt.plot(data3,base,"gx")
    plt.plot(x,norm.pdf(x, mean[2], standDev[2]),"k-",linewidth=1)

    alpha = np.zeros((dataAll.shape[0],3))
    iterations = 1000
    oldMean =0

    for j in range(iterations):
        if math.sqrt(np.dot((mean - oldMean),(mean - oldMean)))< 0.0000001:
            break

        oldMean = copy.deepcopy(mean)
        #E- step
        for k,mean_,standDev_,pi_ in zip(range(3),mean,standDev,pi):
            alpha[:,k] = pi_*norm.pdf(dataAll,mean_,standDev_)
        alpha = np.divide(alpha.T, np.sum(alpha , axis = -1)).T

        ## M- step
        tempMu = np.zeros_like(mean)
        for i in range(3):
            den2 = alpha[:,i].sum()
            mean[i] = (alpha[:,i]*dataAll).sum() / den2
            Num = (alpha[:,i]*(dataAll-mean[i])**2).sum()
            standDev[i] = math.sqrt(Num / den2)
            pi[i] = den2/ dataAll.shape[0]


    plt.figure(5)
    plt.title("FINAL EM RESULTS")
    base = np.zeros_like(data1)
    plt.plot(data1,base,"gx")
    plt.plot(x,norm.pdf(x, mean[0], standDev[0]),"k-",linewidth=1)
    base = np.zeros_like(data2)
    plt.plot(data2,base,"gx")
    plt.plot(x,norm.pdf(x, mean[1], standDev[1]),"k-",linewidth=1)
    base = np.zeros_like(data3)
    plt.plot(data3,base,"gx")
    plt.plot(x,norm.pdf(x, mean[2], standDev[2]),"k-",linewidth=1)
    plt.show()
    print(mean)
    print(standDev)

else:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    import copy
    import math
    from scipy.stats import multivariate_normal
    import random

    N = 99     # Number of training samples
    K = 3       # Number of groups
    D = 2       # Number of dimensions

    data1 =np.random.normal(10, 10,size=(N//3, D))
    data2 =np.random.normal(50, 10,size=(N//3, D))
    data3 =np.random.normal(80, 10,size=(N//3, D))

    x = np.vstack([data1,data2,data3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c,m,x_ in zip(['r','g','y'],['+','^','x'],[data1,data2,data3]):
        plt.plot(x_[:,0], x_[:,1], c=c, marker=m,alpha=0.5)
    ax.set_xlabel('xAxis')
    ax.set_ylabel('yAxis')
    ax.set_zlabel('zAxis')
    plt.title("3D PROJECTION OF 3 CHANNELS")
    plt.show()

    # initilizing gaussians
    initMean = np.random.randint(1, high = 70, size = (K,D)).astype(np.float64)
    initCov =  [(np.random.randint(1,20)*np.eye(D)) for i in range(K)]
    regCov = 1e-6 * np.eye(D)
    initGauss = [multivariate_normal(mean, cov) for mean,cov in zip(initMean,initCov)]
    initPi = np.asarray([1/K for i in range(K)])
    alpha = np.zeros((N,K))
    pdf = np.zeros((N,K))
    log_likelihoods = []

    iterations = 100
    for itera in range(iterations):

        for i,gauss in enumerate(initGauss):
           pdf[:,i] = gauss.pdf(x)

        pixPdf = np.dot(pdf, np.diag(initPi))
        pixPdfSum = np.sum(pixPdf , axis = -1)

        alpha = np.divide(pixPdf.T,
                          pixPdfSum).T

        alphaSum = np.sum(alpha.T , axis = -1)

        mean = np.divide((np.dot(alpha.T,x)).T, alphaSum+1e-8).T

        meanTemp = []
        cov = []
        for i in range(K):
            xSubMean = x-initMean[i, :]
            alphaTemp = np.multiply(alpha[:,i],np.ones((N,D)).T)
            cov.append(np.divide(np.dot(xSubMean.T, np.multiply(alphaTemp.T, xSubMean)), np.sum(alpha[:, i])+1e-8))
            cov[-1] += regCov
        pi = alphaSum/N

        if (np.linalg.norm(mean - initMean) < 1e-4):
            stop = True

        initMean = copy.deepcopy(mean)
        initCov = copy.deepcopy(cov)
        initPi = copy.deepcopy(pi)
        initGauss =  [multivariate_normal(mean_, cov_) for mean_,cov_ in zip(initMean,initCov)]
        log_likelihoods.append(np.log(np.sum([k*multivariate_normal(mean[i],cov[j]).pdf(x) for k,i,j in zip(pi,range(len(mean)),range(len(cov)))])))

    gmm = [multivariate_normal(mean, cov) for mean,cov in zip(initMean,initCov)]

    surface = []
    for i in range(0,255,3):
        for j in range(0,255,3):
            surface.append([i,j])

    surface = np.asarray(surface)
    cpdf = np.zeros(surface.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for gauss,pi,c,m in zip(gmm,initPi,['r','g','y'],['+','^','x']):
        cpdf = np.add(cpdf, pi*gauss.pdf(surface))
        ax.scatter(surface[:,0], surface[:,1], gauss.pdf(surface), c=c, marker=m,alpha=0.1)

    ax.set_xlabel('xAxis')
    ax.set_ylabel('yAxis')
    ax.set_zlabel('zAxis')
    plt.title("2D-EM")
    plt.show()
    #%%
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c,m,x_ in zip(['r','g','y'],['+','^','x'],[data1,data2,data3]):
        plt.plot(x_[:,0], x_[:,1], c=c, marker=m,alpha=0.5)
    ax.scatter(surface[:,0], surface[:,1], cpdf, c='k', marker='.',alpha=0.2)
    ax.set_xlabel('xAxis')
    ax.set_ylabel('yAxis')
    ax.set_zlabel('zAxis')
    plt.title("2D-EM")
    plt.show()
    print(mean)
    print(cov)
