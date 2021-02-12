import cv2
import numpy as np
import glob
from scipy.stats import multivariate_normal
import copy
out = cv2.VideoWriter('3D_GMM.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (640, 480))
DATASET = "DETECTBUOY-FRAMES/Data"

def Ellipse_Fit(mask):
    processed = mask.astype(np.uint8)
    processed = cv2.GaussianBlur(processed, (5, 5), cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(processed, 60, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 300 and cv2.contourArea(cnt) < 5000:
            ellipses.append(cv2.fitEllipse(cnt))
    outEllipse = []
    for ell in ellipses:
        (x, y), (MA, ma), angle = ell
        if abs(MA / ma - 1) < 0.3:
            outEllipse.append(ell)
    return outEllipse


def High_PDF(prob, threshold):
    p = prob.reshape((prob.shape[0] * prob.shape[1], prob.shape[2]))
    q = np.multiply(p, p > threshold)
    b = np.multiply(q > 0, np.equal(q, np.max(q, axis=-1, keepdims=True))) * 255
    c = b.reshape((prob.shape[0], prob.shape[1], prob.shape[2]))
    return c

def Water_Mask(frame):
    # For redBuoy1
    mean = np.array([80.27603646, 141.43706643, 253.22644464])
    cov = np.array([[190.60613704, 201.66921469, -5.62641894],
                    [201.66921469, 340.80624709, -14.2263423],
                    [-5.62641894, -14.2263423, 3.51000389]])
    P_RB1 = multivariate_normal.pdf(frame, mean, cov)

    # For redBuoy2
    mean = np.array([129.75146712, 187.0247840, 232.87476706])
    cov = np.array([[792.3089489, 966.06181438, -76.63443504],
                    [966.06181438, 1358.97343543, -15.6558208],
                    [-76.63443504, -15.6558208, 274.29810684]])
    P_RB2 = multivariate_normal.pdf(frame, mean, cov)

    # For redBuoy3
    mean = np.array([117.81710669, 204.2309085, 239.41048976])
    cov = np.array([[443.75427994, 518.28342899, -139.95097112],
                    [518.28342899, 707.05237291, -187.05091184],
                    [-139.95097112, -187.05091184, 64.27720605]])
    P_RB3 = multivariate_normal.pdf(frame, mean, cov)

    P_RB = P_RB1 + P_RB2 + P_RB3

    # For Green1
    mean = np.array([112.05003011, 183.18656764, 103.53271839])
    cov = np.array([[98.18729895, 128.48175019, 111.23031125],
                    [128.48175019, 372.47086917, 237.17047113],
                    [111.23031125, 237.17047113, 230.78640153]])
    P_GB1 = multivariate_normal.pdf(frame, mean, cov)

    # For Green2
    mean = np.array([125.22320558, 229.46544678, 142.17248589])
    cov = np.array([[83.42004155, 109.12603316, 133.04099339],
                    [109.12603316, 181.75339967, 209.44426981],
                    [133.04099339, 209.44426981, 280.21373779]])
    P_GB2 = multivariate_normal.pdf(frame, mean, cov)

    # For Green3
    mean = np.array([150.32076907, 239.42616469, 187.56685088])
    cov = np.array([[296.42463121, 109.06686387, 351.389052],
                    [109.06686387, 138.29429843, 172.87515629],
                    [351.389052, 172.87515629, 653.94501523]])
    P_GB3 = multivariate_normal.pdf(frame, mean, cov)

    P_GB = P_GB1 + P_GB2 + P_GB3

    # For yellowBuoy
    mean = np.array([93.18674196, 204.10273852, 208.83574233])
    cov = np.array([[325.95744462, 14.78707018, -304.72169773],
                    [14.78707018, 161.85807802, 267.4821683],
                    [-304.72169773, 267.4821683, 890.87026603]])
    P_YB = multivariate_normal.pdf(frame, mean, cov)

    # For Water1
    mean = np.array([154.242466 ,228.26091272,233.45074722])
    cov = np.array([[59.2038326 , 46.17327671,  5.3503438 ],
                   [46.17327671, 58.66903207, -7.51014766],
                   [ 5.3503438 , -7.51014766, 26.28058457]])
    P_W1 = multivariate_normal.pdf(frame, mean, cov)

    mean = np.array([141.96297332 ,204.83155696,220.47708726])
    cov = np.array([[100.70632783, 148.60410607,  59.9378063 ],
                   [148.60410607, 320.22102525, 129.64470878],
                   [ 59.9378063 , 129.64470878, 121.25904618]])
    P_W2 = multivariate_normal.pdf(frame, mean, cov)

    mean = np.array([178.2135104  ,238.03114502 ,180.63696875])
    cov = np.array([[ 44.16861721,  46.21022285,  68.88757629],
                   [ 46.21022285,  58.90147946,  78.51143783],
                   [ 68.88757629,  78.51143783, 203.85445566]])
    P_W3 = multivariate_normal.pdf(frame, mean, cov)

    P_W = P_W1 + P_W2 + P_W3

    prob = np.zeros((frame.shape[0], frame.shape[1], 4))

    prob[:, :, 0] = P_RB
    prob[:, :, 1] = P_GB
    prob[:, :, 2] = P_YB
    prob[:, :, 3] = P_W * 0.99

    # best results with Multiply
    RGY_Buoy = High_PDF(prob, 1e-15)  # -15
    return RGY_Buoy

def Buoy_data(waterRemoved):
    # For redBuoy1
    mean = np.array([129.75151074, 187.02495822, 232.87487513])
    cov = np.array([[792.30842907, 966.0620035, -76.63515958],
                    [966.0620035, 1358.97477086, -15.65802897],
                    [-76.63515958, -15.65802897, 274.29390402]])
    P_RB1 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For redBuoy2
    mean = np.array([117.81699529, 204.23082796, 239.41051339])
    cov = np.array([[443.75320996, 518.2835338, -139.95105276],
                    [518.2835338, 707.05318175, -187.05121695],
                    [-139.95105276, -187.05121695, 64.27726249]])
    P_RB2 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For redBuoy3
    mean = np.array([81.53413865, 141.57207486, 253.14210245])
    cov = np.array([[228.92875888, 224.1567059, -7.02999134],
                    [224.1567059, 339.10305449, -13.59245238],
                    [-7.02999134, -13.59245238, 3.91363665]])
    P_RB3 = multivariate_normal.pdf(waterRemoved, mean, cov)

    PiRb = np.array([0.15838274, 0.38113269, 0.44139788])
    P_RB = PiRb[0] * P_RB1 + PiRb[1] * P_RB2 + PiRb[2] * P_RB3

    # For Green1
    mean = np.array([110.15586103, 177.988079, 97.8360865])
    cov = np.array([[82.84302567, 106.35540435, 74.22384909],
                    [106.35540435, 306.33086617, 154.3897207],
                    [74.22384909, 154.3897207, 118.64202382]])
    P_GB1 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For Green2
    mean = np.array([124.00448114, 217.39861905, 136.44552769])
    cov = np.array([[135.27527716, 132.43005772, 186.54968698],
                    [132.43005772, 361.10595221, 281.7120668],
                    [186.54968698, 281.7120668, 375.55342302]])
    P_GB2 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For Green3
    mean = np.array([152.97075593, 244.63284543, 194.2491698])
    cov = np.array([[269.37418864, 37.51788466, 286.85356749],
                    [37.51788466, 38.57928137, 14.06820397],
                    [286.85356749, 14.06820397, 491.56890665]])
    P_GB3 = multivariate_normal.pdf(waterRemoved, mean, cov)

    PiGb = np.array([0.39978126, 0.38033716, 0.19886462])

    P_GB = PiGb[0] * P_GB1 + PiGb[1] * P_GB2 + PiGb[2] * P_GB3

    # For yellowBuoy1
    mean = np.array([124.48956165, 235.49979435, 232.22955126])
    cov = np.array([[1165.98834055, 180.00433825, -59.25367115],
                    [180.00433825, 78.85588687, 20.33064827],
                    [-59.25367115, 20.33064827, 81.66227936]])
    P_YB1 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For yellowBuoy
    mean = np.array([93.18674196, 204.10273852, 208.83574233])
    cov = np.array([[325.95744462, 14.78707018, -304.72169773],
                    [14.78707018, 161.85807802, 267.4821683],
                    [-304.72169773, 267.4821683, 890.87026603]])
    P_YB2 = multivariate_normal.pdf(waterRemoved, mean, cov)

    # For yellowBuoy
    mean = np.array([138.56180468, 240.07565167, 229.07810767])
    cov = np.array([[775.88598663, -42.21694591, -40.46084514],
                    [-42.21694591, 4.60254418, 2.08209706],
                    [-40.46084514, 2.08209706, 6.96561565]])
    P_YB3 = multivariate_normal.pdf(waterRemoved, mean, cov)

    PiYb = np.array([0.26255614, 0.2175131, 0.50246477])

    P_YB = PiYb[0] * P_YB1 + PiYb[1] * P_YB2 + PiYb[2] * P_YB3

    prob = np.zeros((frame.shape[0], frame.shape[1], 3))

    prob[:, :, 0] = P_RB
    prob[:, :, 1] = P_GB
    prob[:, :, 2] = P_YB

    RGY_Buoy_2 = High_PDF(prob, 1e-6)  # -20
    return RGY_Buoy_2

def Draw_Ellipse(RGY_Buoy_2,Image_Input):
    ellipseR = Ellipse_Fit(RGY_Buoy_2[:, :, 0].astype(np.uint8))
    Image_Input_1 = copy.deepcopy(Image_Input)
    for ell in ellipseR:
        cv2.ellipse(Image_Input_1, ell, (0, 0, 255), 5)

    ellipseG = Ellipse_Fit(RGY_Buoy_2[:, :, 1].astype(np.uint8))
    for ell in ellipseG:
        cv2.ellipse(Image_Input_1, ell, (0, 255, 0), 5)

    ellipseY = Ellipse_Fit(RGY_Buoy_2[:, :, 2].astype(np.uint8))
    for ell in ellipseY:
        cv2.ellipse(Image_Input_1, ell, (0, 255, 255), 5)
    return Image_Input_1


for file in glob.glob(f"{DATASET}/*.jpg"):
    Image_Input = cv2.imread(file)
    frame = np.zeros((Image_Input.shape[0], Image_Input.shape[1], 3))
    frame[:, :, 0] = Image_Input[:, :, 0]
    frame[:, :, 1] = Image_Input[:, :, 1]
    frame[:, :, 2] = Image_Input[:, :, 2]
    ## Order of Probabilities - green, red
    RGY_Buoy = Water_Mask(frame)
    Water_Remove = RGY_Buoy[:, :, 3].astype(np.int8)
    Water_Remove = cv2.bitwise_not(Water_Remove)
    waterRemoved = cv2.bitwise_and(Image_Input, Image_Input, mask=Water_Remove)
    # cv2.imshow("WATERMASK",waterRemoved)
    RGY_Buoy_2 = Buoy_data(waterRemoved)
    redBuoySegement = cv2.bitwise_and(Image_Input, Image_Input, mask=RGY_Buoy_2[:, :, 0].astype(np.int8))
    greenBuoySegment = cv2.bitwise_and(Image_Input, Image_Input, mask=RGY_Buoy_2[:, :, 1].astype(np.int8))
    yellowBuoySegment = cv2.bitwise_and(Image_Input, Image_Input, mask=RGY_Buoy_2[:, :, 2].astype(np.int8))
    # cv2.imshow("R-BUOY",redBuoySegement)
    # cv2.imshow("G-BUOY",greenBuoySegment)
    # cv2.imshow("Y-BUOY",yellowBuoySegment)
    Image_Input_1 = Draw_Ellipse(RGY_Buoy_2, Image_Input)
    out.write(Image_Input_1)
    cv2.imshow("ALL-BUOY-DETECT", Image_Input_1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()