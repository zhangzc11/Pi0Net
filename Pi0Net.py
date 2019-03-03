#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, roc_curve
import root_pandas as rp
import numpy as np
import ROOT as root
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
#from keras import regularizers

variables_array = ["STr2_ehitG_rec", "STr2_overlapG_rec", "STr2_gapG_rec"]
variables_1d = ["STr2_enG_rec", "STr2_enG_true"]

inputFileName = 'data/photonNtuple.root'
inputFile = root.TFile(inputFileName)

tree_train_gamma1 = inputFile.Get("Tree_Optim_train_gamma1")
NEntries_train_gamma1 = tree_train_gamma1.GetEntries()
print "NEntries of gamma1: "+str(NEntries_train_gamma1)
df_train_gamma1 = rp.read_root(inputFileName, key="Tree_Optim_train_gamma1", columns=variables_array+variables_1d, flatten=variables_array)
arr_df_train_gamma1 = df_train_gamma1.values
X_train_gamma1 = np.zeros((NEntries_train_gamma1, 3, 3, 3))
Y_train_gamma1 = np.zeros(NEntries_train_gamma1)

tree_train_gamma2 = inputFile.Get("Tree_Optim_train_gamma2")
NEntries_train_gamma2 = tree_train_gamma2.GetEntries()
print "NEntries of gamma2: "+str(NEntries_train_gamma2)
df_train_gamma2 = rp.read_root(inputFileName, key="Tree_Optim_train_gamma2", columns=variables_array+variables_1d, flatten=variables_array)
arr_df_train_gamma2 = df_train_gamma2.values
X_train_gamma2 = np.zeros((NEntries_train_gamma2, 3, 3, 3))
Y_train_gamma2 = np.zeros(NEntries_train_gamma2)


for idx in range(NEntries_train_gamma1):
	temp = np.array([
		[
		  [arr_df_train_gamma1[idx*9+8][0], arr_df_train_gamma1[idx*9+7][0], arr_df_train_gamma1[idx*9+6][0]],
		  [arr_df_train_gamma1[idx*9+5][0], arr_df_train_gamma1[idx*9+4][0], arr_df_train_gamma1[idx*9+3][0]],
		  [arr_df_train_gamma1[idx*9+2][0], arr_df_train_gamma1[idx*9+1][0], arr_df_train_gamma1[idx*9+0][0]]
		],
		
		[
		  [arr_df_train_gamma1[idx*9+8][1], arr_df_train_gamma1[idx*9+7][1], arr_df_train_gamma1[idx*9+6][1]],
		  [arr_df_train_gamma1[idx*9+5][1], arr_df_train_gamma1[idx*9+4][1], arr_df_train_gamma1[idx*9+3][1]],
		  [arr_df_train_gamma1[idx*9+2][1], arr_df_train_gamma1[idx*9+1][1], arr_df_train_gamma1[idx*9+0][1]]
		],
		[
		  [arr_df_train_gamma1[idx*9+8][2], arr_df_train_gamma1[idx*9+7][2], arr_df_train_gamma1[idx*9+6][2]],
		  [arr_df_train_gamma1[idx*9+5][2], arr_df_train_gamma1[idx*9+4][2], arr_df_train_gamma1[idx*9+3][2]],
		  [arr_df_train_gamma1[idx*9+2][2], arr_df_train_gamma1[idx*9+1][2], arr_df_train_gamma1[idx*9+0][2]]
		]
	      ])
	X_train_gamma1[idx] = np.copy(np.rollaxis(temp, 0, 3))
	Y_train_gamma1[idx] = arr_df_train_gamma1[idx*9][4]


for idx in range(NEntries_train_gamma2):
	temp = np.array([
		[
		  [arr_df_train_gamma2[idx*9+8][0], arr_df_train_gamma2[idx*9+7][0], arr_df_train_gamma2[idx*9+6][0]],
		  [arr_df_train_gamma2[idx*9+5][0], arr_df_train_gamma2[idx*9+4][0], arr_df_train_gamma2[idx*9+3][0]],
		  [arr_df_train_gamma2[idx*9+2][0], arr_df_train_gamma2[idx*9+1][0], arr_df_train_gamma2[idx*9+0][0]]
		],
		
		[
		  [arr_df_train_gamma2[idx*9+8][1], arr_df_train_gamma2[idx*9+7][1], arr_df_train_gamma2[idx*9+6][1]],
		  [arr_df_train_gamma2[idx*9+5][1], arr_df_train_gamma2[idx*9+4][1], arr_df_train_gamma2[idx*9+3][1]],
		  [arr_df_train_gamma2[idx*9+2][1], arr_df_train_gamma2[idx*9+1][1], arr_df_train_gamma2[idx*9+0][1]]
		],
		[
		  [arr_df_train_gamma2[idx*9+8][2], arr_df_train_gamma2[idx*9+7][2], arr_df_train_gamma2[idx*9+6][2]],
		  [arr_df_train_gamma2[idx*9+5][2], arr_df_train_gamma2[idx*9+4][2], arr_df_train_gamma2[idx*9+3][2]],
		  [arr_df_train_gamma2[idx*9+2][2], arr_df_train_gamma2[idx*9+1][2], arr_df_train_gamma2[idx*9+0][2]]
		]
	      ])
	X_train_gamma2[idx] = np.copy(np.rollaxis(temp, 0, 3))
	Y_train_gamma2[idx] = arr_df_train_gamma2[idx*9][4]



for idx in range(5):
	print "event "+str(idx)
	print "gamma1 hit map:"
	print X_train_gamma1[idx]
	print "gamma1 total rec E = "+str(np.sum(X_train_gamma1[idx,:,:,0]))
	print "gamma1 true E = "+str(Y_train_gamma1[idx])

	print "gamma2 hit map:"
	print X_train_gamma2[idx]
	print "gamma2 total rec E = "+str(np.sum(X_train_gamma2[idx,:,:,0]))
	print "gamma2 true E = "+str(Y_train_gamma2[idx])


