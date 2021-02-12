import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import glob
import copy

MeanR = []
VarianceR = []
MeanG = []
VarianceG = []
MeanY = []
VarianceY = []
MeanW = []
VarianceW = []
ChannelRed = np.empty(1)
ChannelBlue = np.empty(1)
ChannelGreen = np.empty(1)
AllRed = []
AllBlue = []
AllGreen = []
ImgRed = cv2.imread('TrainingImages/Red.png')
ImgGreen = cv2.imread('TrainingImages/Green.png')
ImgYellow = cv2.imread('TrainingImages/Yellow.png')
ImgWater = cv2.imread('TrainingImages/Water2.png')
roiR = np.nonzero(ImgRed)
roiG = np.nonzero(ImgGreen)
roiY = np.nonzero(ImgYellow)
roiW = np.nonzero(ImgWater)
out = cv2.VideoWriter('1D_Gaussian.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (640,480))
testFrames = "DETECTBUOY-FRAMES/Data"
x = np.linspace(0,255, num=100)

def High_PDF(prob,threshold):
    p = prob.reshape((prob.shape[0]*prob.shape[1],prob.shape[2]))
    q = np.multiply(p,p>threshold)
    b = np.multiply(q>0, np.equal(q, np.max(q, axis=-1, keepdims = True)))*255
    c = b.reshape((prob.shape[0],prob.shape[1],prob.shape[2]))
    return c

def Ellipse_Fit(mask):
    processed = mask.astype(np.uint8)
    processed = cv2.blur(processed, (5, 5))
    ret, thresh = cv2.threshold(processed, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 250 and cv2.contourArea(cnt) < 5000:
            ellipses.append( cv2.fitEllipse(cnt))
    outEllipse = []
    for ell in ellipses:
        (x,y),(MA,ma),angle = ell
        if abs(MA/ma-1) <0.2:
            outEllipse.append(ell)
    return outEllipse

def Get_RGB_Channels(ChannelRed,ChannelGreen,ChannelBlue):
    for r, img in zip([roiR,roiG,roiY,roiW],[ImgRed,ImgGreen,ImgYellow,ImgWater]):
        ChannelRed = np.hstack((ChannelRed,img[r[0],r[1],2]))
        ChannelGreen = np.hstack((ChannelGreen,img[r[0],r[1],1]))
        ChannelBlue = np.hstack((ChannelBlue,img[r[0],r[1],0]))
        AllRed.append(ChannelRed)
        AllBlue.append(ChannelBlue)
        AllGreen.append(ChannelGreen)
    return AllRed,AllBlue,AllGreen

def GetParameters(AllRed,AllBlue,AllGreen):
    print("|-------------------RED BUOY-------------------|")
    # plt.figure(1)
    MeanR.append(np.mean(AllBlue[0]))
    VarianceR.append(np.var(AllBlue[0]))
    # plt.subplot(421)
    # plt.title("RED BUOY",color='r')
    # plt.hist(AllBlue[0], 256, [0, 256], color='b')
    # plt.subplot(422)
    # plt.plot(x, stats.norm.pdf(x, MeanR[0], math.sqrt(VarianceR[0])), 'b-')
    MeanR.append(np.mean(AllGreen[0]))
    VarianceR.append(np.var(AllGreen[0]))
    # plt.subplot(423)
    # plt.hist(AllGreen[0], 256, [0, 256], color='g')
    # plt.subplot(424)
    # plt.plot(x, stats.norm.pdf(x, MeanR[1], math.sqrt(VarianceR[1])), 'g-')
    MeanR.append(np.mean(AllRed[0]))
    VarianceR.append(np.var(AllRed[0]))
    # plt.subplot(425)
    # plt.hist(AllRed[0], 256, [0, 256], color='r')
    # plt.subplot(426)
    # plt.plot(x, stats.norm.pdf(x, MeanR[2], math.sqrt(VarianceR[2])), 'r-')
    print("MeanR: ", MeanR)
    print("VarianceR: ", VarianceR)
    
    print("|-------------------GREEN BUOY-------------------|")
    # plt.figure(2)
    MeanG.append(np.mean(AllBlue[1]))
    VarianceG.append(np.var(AllBlue[1]))
    # plt.subplot(421)
    # plt.title("GREEN BUOY",color='g')
    # plt.hist(AllBlue[1], 256, [0, 256], color='b')
    # plt.subplot(422)
    # plt.plot(x, stats.norm.pdf(x, MeanG[0], math.sqrt(VarianceG[0])), 'b-')
    MeanG.append(np.mean(AllGreen[1]))
    VarianceG.append(np.var(AllGreen[1]))
    # plt.subplot(423)
    # plt.hist(AllGreen[1], 256, [0, 256], color='g')
    # plt.subplot(424)
    # plt.plot(x, stats.norm.pdf(x, MeanG[1], math.sqrt(VarianceG[1])), 'g-')
    MeanG.append(np.mean(AllRed[1]))
    VarianceG.append(np.var(AllRed[1]))
    # plt.subplot(425)
    # plt.hist(AllRed[1], 256, [0, 256], color='r')
    # plt.subplot(426)
    # plt.plot(x, stats.norm.pdf(x, MeanG[2], math.sqrt(VarianceG[2])), 'r-')
    print("MeanG: ", MeanG)
    print("VarianceG: ", VarianceG)

    print("|-------------------YELLOW BUOY-------------------|")
    # plt.figure(3)
    MeanY.append(np.mean(AllBlue[2]))
    VarianceY.append(np.var(AllBlue[2]))
    # plt.subplot(421)
    # plt.title("YELLOW BUOY",color='y')
    # plt.hist(AllBlue[2], 256, [0, 256],  color='b')
    # plt.subplot(422)
    # plt.plot(x, stats.norm.pdf(x, MeanY[0], math.sqrt(VarianceY[0])), 'b-')
    MeanY.append(np.mean(AllGreen[2]))
    VarianceY.append(np.var(AllGreen[2]))
    # plt.subplot(423)
    # plt.hist(AllGreen[2], 256, [0, 256], color='g')
    # plt.subplot(424)
    # plt.plot(x, stats.norm.pdf(x, MeanY[1], math.sqrt(VarianceY[1])), 'g-')
    MeanY.append(np.mean(AllRed[2]))
    VarianceY.append(np.var(AllRed[2]))
    # plt.subplot(425)
    # plt.hist(AllRed[2], 256, [0, 256], color='r')
    # plt.subplot(426)
    # plt.plot(x, stats.norm.pdf(x, MeanY[2], math.sqrt(VarianceY[2])), 'r-')
    print("MeanY: ", MeanY)
    print("VarianceY: ", VarianceY)
    
    print("|-------------------WATER-------------------|")
    MeanW.append(np.mean(AllBlue[3]))
    VarianceW.append(np.var(AllBlue[3]))
    MeanW.append(np.mean(AllGreen[3]))
    VarianceW.append(np.var(AllGreen[3]))
    MeanW.append(np.mean(AllRed[3]))
    VarianceW.append(np.var(AllRed[3]))
    print("MeanY: ", MeanW)
    print("VarianceY: ", MeanW)
    # print("COMPARING ALL RGB CHANELS FOR EACH BUOY!!")
    # plt.figure(4)
    # plt.title("BLUE CHANNEL",color='b')
    # plt.plot(x, stats.norm.pdf(x, MeanR[0], math.sqrt(VarianceR[0])),'bx',label = 'Red Buoy')
    # plt.plot(x, stats.norm.pdf(x, MeanG[0], math.sqrt(VarianceG[0])),'b-',label = 'Green Buoy')
    # plt.plot(x, stats.norm.pdf(x, MeanY[0], math.sqrt(VarianceY[0])),'b*',label = 'Yellow Buoy')
    # plt.legend()
    # plt.figure(5)
    # plt.title("GREEN CHANNEL",color='g')
    # plt.plot(x, stats.norm.pdf(x, MeanR[1], math.sqrt(VarianceR[1])),'gx',label = 'Red Buoy')
    # plt.plot(x, stats.norm.pdf(x, MeanG[1], math.sqrt(VarianceG[1])),'g-',label = 'Green Buoy')
    # plt.plot(x, stats.norm.pdf(x, MeanY[1], math.sqrt(VarianceY[1])),'g*',label = 'Yellow Buoy')
    # plt.legend()
    # plt.figure(6)
    # plt.title("RED CHANNEL",color='r')
    # plt.plot(x, stats.norm.pdf(x, MeanR[2], math.sqrt(VarianceR[2])),'rx',label = 'Red Buoy')
    # plt.plot(x, stats.norm.pdf(x, MeanG[2], math.sqrt(VarianceG[2])),'r-',label = 'Green Buoy')
    # plt.plot(x, stats.norm.pdf(x, MeanY[2], math.sqrt(VarianceY[2])),'r*',label = 'Yellow Buoy')
    # plt.legend()
    return MeanG,MeanR,MeanW,MeanY,VarianceG,VarianceR,VarianceW,VarianceY

def Detect(MeanG,MeanR,MeanW,MeanY,VarianceG,VarianceR,VarianceW,VarianceY):
    for file in glob.glob(f"{testFrames}/*.jpg"):
        frame = cv2.imread(file)

        # For redBuoy
        ProbRB = np.zeros((frame.shape[0], frame.shape[1], 2))
        # Prob of green channel
        ProbRB[:, :, 0] = stats.norm.pdf(frame[:, :, 1], MeanR[1], math.sqrt(VarianceR[1]))
        # Prob of red channel
        ProbRB[:, :, 1] = stats.norm.pdf(frame[:, :, 2], MeanR[2], math.sqrt(VarianceR[2]))

        # For greenBuoy
        ProbGB = np.zeros((frame.shape[0], frame.shape[1], 3))
        # Prob of green channel
        ProbGB[:, :, 0] = stats.norm.pdf(frame[:, :, 0], MeanG[0], math.sqrt(VarianceG[0]))
        # Prob of red channel
        ProbGB[:, :, 1] = stats.norm.pdf(frame[:, :, 1], MeanG[1], math.sqrt(VarianceG[1]))
        ProbGB[:, :, 2] = stats.norm.pdf(frame[:, :, 2], MeanG[2], math.sqrt(VarianceG[2]))

        # For yellowBuoy
        ProbYB = np.zeros((frame.shape[0], frame.shape[1], 2))
        # Prob of green channel
        ProbYB[:, :, 0] = stats.norm.pdf(frame[:, :, 1], MeanY[1], math.sqrt(VarianceY[1]))
        # Prob of red channel
        ProbYB[:, :, 1] = stats.norm.pdf(frame[:, :, 2], MeanY[2], math.sqrt(VarianceY[2]))

        # For Water
        ProbW3 = np.zeros((frame.shape[0], frame.shape[1], 3))
        # Prob of green channel
        ProbW3[:, :, 0] = stats.norm.pdf(frame[:, :, 0], MeanW[0], math.sqrt(VarianceW[0]))
        # Prob of red channel
        ProbW3[:, :, 1] = stats.norm.pdf(frame[:, :, 1], MeanW[1], math.sqrt(VarianceW[1]))

        ProbRB_final = ProbRB[:, :, 0] * ProbRB[:, :, 1]
        ProbGB_final = ProbGB[:, :, 0] * ProbGB[:, :, 1]
        ProbYB_final = ProbYB[:, :, 0] * ProbYB[:, :, 1]
        ProbW3_final = ProbW3[:, :, 0] * ProbW3[:, :, 1]

        prob = np.zeros((frame.shape[0], frame.shape[1], 4))

        prob[:, :, 0] = ProbRB_final
        prob[:, :, 1] = ProbGB_final
        prob[:, :, 2] = ProbYB_final
        prob[:, :, 3] = ProbW3_final

        # Multiplying
        rgy = High_PDF(prob, 1e-5)
        redBuoySegement = cv2.bitwise_and(frame, frame, mask=rgy[:, :, 0].astype(np.int8))
        greenBuoySegment = cv2.bitwise_and(frame, frame, mask=rgy[:, :, 1].astype(np.int8))
        yellowBuoySegment = cv2.bitwise_and(frame, frame, mask=rgy[:, :, 2].astype(np.int8))

        # cv2.imshow("REDBUOY-MASK", redBuoySegement)
        # cv2.imshow("GREENBUOY-MASK", greenBuoySegment)
        # cv2.imshow("YELLOWBUOY-MASK", yellowBuoySegment)

        ellipseR = Ellipse_Fit(rgy[:, :, 0].astype(np.uint8))
        Img_ = copy.deepcopy(frame)
        for ell in ellipseR:
            cv2.ellipse(Img_, ell, (0, 0, 255), 3)

        ellipseG = Ellipse_Fit(rgy[:, :, 1].astype(np.uint8))
        for ell in ellipseG:
            cv2.ellipse(Img_, ell, (0, 255, 0), 3)

        ellipseY = Ellipse_Fit(rgy[:, :, 2].astype(np.uint8))
        for ell in ellipseY:
            cv2.ellipse(Img_, ell, (0, 255, 255), 3)

        out.write(Img_)
        cv2.imshow("DETECTBUOY", Img_)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

All_Red,All_Blue,All_Green = Get_RGB_Channels(ChannelRed,ChannelGreen,ChannelBlue)
Mean_G,Mean_R,Mean_W,Mean_Y,Variance_G,Variance_R,Variance_W,Variance_Y = GetParameters(All_Red,All_Blue,All_Green)
print("Please Wait Generating Video")
Detect(Mean_G,Mean_R,Mean_W,Mean_Y,Variance_G,Variance_R,Variance_W,Variance_Y)
print("Generating Video Successful")
out.release()
cv2.destroyAllWindows()

