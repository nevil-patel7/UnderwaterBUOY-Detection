Directory Structure-

	akurhade_hw3-
		|--Code-
		|  |-DETECTBUOY-FRAMES-
		|  |	|-Data		#Data frames
		|  |-TrainingImages-	#Training images for buoys and water
		|  |-1D_GAUSSIAN.py	#generated 1-D Gaussian
		|  |-3D_GMM.py		# Expands 1-D Gaussian to 2-d plane
		|  |-EM_Algo.py		# Expectation Maximization
		|  | GETDATA_3DGmm.py	#Get Mean and C-variance
		|--Report.pdf
		|--README.md

System Requirements-
	Python (v3.0.x or later)

Libraries needed-
	1) OpenCV (opencv-python)
	2) Numpy
	3) Scipy 
	4) matplotlib

Running Instructions-
Script '1D_Gaussian.py'-
	Requirements-
		1) Source directory path for Training images of buoys (eg. TrainingImages/Red.png)
		2) Source Directory path for data frames (DETECTBUOY-FRAMES/Data)
Script '3D_GMM.py'-
	Requirements-
		1) Dataset source path (DETECTBUOY-FRAMES/Data)
Script 'EM_ALgo.py'
	Requirements-
		Input- Choose between 1-d and 2-d EM
			 Enter 0 -> 1-D EM 
			 or
			       1 -> 2-D EM
Script 'GETDATA_3DGMM'-
Requirements-
		1) Source directory path for Training images of buoys (eg. TrainingImages/Red.png)
		2) Source Directory path for data frames (DETECTBUOY-FRAMES/Data)




			