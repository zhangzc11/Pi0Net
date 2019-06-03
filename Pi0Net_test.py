
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import root_pandas as rp
import numpy as np
import ROOT as root
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Concatenate, concatenate
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input
from keras import regularizers
from scipy.optimize import curve_fit
from xgboost import XGBRegressor
import keras.backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants


# In[2]:


def get_array_from_rootfile(inputFileName, treename):
    inputFile = root.TFile(inputFileName)
    variables_array = ["STr2_ehitG_rec", "STr2_overlapG_rec", "STr2_gapG_rec"]
    variables_1d = ["STr2_enG_rec", "STr2_enG_true", "STr2_iEtaiX", "STr2_iPhiiY", 
                    "STr2_Eta", "STr2_phi", "STr2_Nxtal", "STr2_S1S9", "STr2_S4S9"]
    tree = inputFile.Get(treename)
    NEntries = tree.GetEntries()
    print("Reading tree "+treename+", NEntries = "+str(NEntries))
    dfp = rp.read_root(inputFileName, key=treename, columns=variables_array+variables_1d, flatten=variables_array)
    df = dfp.reindex(columns=variables_array+variables_1d)
    arr_df = df.values
    X = np.zeros((NEntries, 3, 3, 5))
    X_dense = np.zeros((NEntries, 6))#Eraw, eta, phi, Nxtal, S1S9, S4S9
    Y = np.zeros(NEntries)
    Y_raw = np.zeros(NEntries)
    for idx in range(NEntries):
        if idx%(int(NEntries/10)) == 0:
            print("processing event %d out of %d"%(idx, NEntries))
        temp = np.array([
                [
                  [arr_df[idx*9+8][0], arr_df[idx*9+7][0], arr_df[idx*9+6][0]],
                  [arr_df[idx*9+5][0], arr_df[idx*9+4][0], arr_df[idx*9+3][0]],
                  [arr_df[idx*9+2][0], arr_df[idx*9+1][0], arr_df[idx*9+0][0]]
                ],

                [
                  [arr_df[idx*9+8][1], arr_df[idx*9+7][1], arr_df[idx*9+6][1]],
                  [arr_df[idx*9+5][1], arr_df[idx*9+4][1], arr_df[idx*9+3][1]],
                  [arr_df[idx*9+2][1], arr_df[idx*9+1][1], arr_df[idx*9+0][1]]
                ],
                [
                  [arr_df[idx*9+8][2], arr_df[idx*9+7][2], arr_df[idx*9+6][2]],
                  [arr_df[idx*9+5][2], arr_df[idx*9+4][2], arr_df[idx*9+3][2]],
                  [arr_df[idx*9+2][2], arr_df[idx*9+1][2], arr_df[idx*9+0][2]]
                ],
                [
                  [arr_df[idx*9+8][5]-1, arr_df[idx*9+7][5], arr_df[idx*9+6][5]+1],
                  [arr_df[idx*9+5][5]-1, arr_df[idx*9+4][5], arr_df[idx*9+3][5]+1],
                  [arr_df[idx*9+2][5]-1, arr_df[idx*9+1][5], arr_df[idx*9+0][5]+1]
                ],
                [
                  [arr_df[idx*9+8][6]+1, arr_df[idx*9+7][6]+1, arr_df[idx*9+6][6]+1],
                  [arr_df[idx*9+5][6], arr_df[idx*9+4][6], arr_df[idx*9+3][6]],
                  [arr_df[idx*9+2][6]-1, arr_df[idx*9+1][6]-1, arr_df[idx*9+0][6]-1]
                ]
              ])
        X[idx] = np.copy(np.rollaxis(temp, 0, 3))
        Y[idx] = arr_df[idx*9][4]
        X_dense[idx][0] = arr_df[idx*9][3]
        X_dense[idx][1] = arr_df[idx*9][7]
        X_dense[idx][2] = arr_df[idx*9][8]
        X_dense[idx][3] = arr_df[idx*9][9]
        X_dense[idx][4] = arr_df[idx*9][10]
        X_dense[idx][5] = arr_df[idx*9][11]
        Y_raw[idx] = arr_df[idx*9][3]
        
    return X, X_dense, Y, Y_raw


# In[3]:


X_test_gamma1, X_test_dense_gamma1, Y_test_gamma1, Y_raw_test_gamma1 = get_array_from_rootfile("data/small/photonNtuple_EB.root", "Tree_Optim_test_gamma1")


# In[4]:


from keras.models import load_model, model_from_json
K.set_learning_phase(0)
json_file = open('model_Pi0Net_CNN_dense_E_1000epoch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('weights_Pi0Net_CNN_dense_E_1000epoch.h5')
Y_predict_test_gamma1 = loaded_model.predict([X_test_gamma1[:,:,:,:3], X_test_dense_gamma1]).flatten()
print(Y_predict_test_gamma1[:10])


# In[5]:


def effSigma(a):
    #input: an array
    #output: the smallest width which contains 68.3% of the distribution
    rlim = 0.683
    quantile_scan = np.arange(0.0, 1-rlim, 0.001)
    min_width = a.max() - a.min()
    for q in quantile_scan:
        aLeft = np.quantile(a, q)
        aRight = np.quantile(a, q+rlim)
        if aRight - aLeft < min_width:
            min_width = aRight - aLeft
    return min_width


# In[6]:


def plot_E_over_Etrue(Y_true, Y_raw, Y_cor, plotname):
    plt.clf()
    Y_raw_over_true = []
    Y_cor_over_true = []
    Y_raw_over_true = np.divide(Y_raw, Y_true)
    Y_cor_over_true = np.divide(Y_cor, Y_true)
    std_raw = np.std(Y_raw_over_true)
    std_cor = np.std(Y_cor_over_true)
    eff_sig_raw = effSigma(Y_raw_over_true)
    eff_sig_cor = effSigma(Y_cor_over_true)
    plt.hist(Y_cor_over_true, bins=200, range=(0,1.68), alpha=0.5, label='$E_{cor}$/$E_{true}$, $\sigma_{eff}$=%.2f'%eff_sig_cor)
    plt.hist(Y_raw_over_true, bins=200, range=(0,1.68), alpha=0.5, label='$E_{raw}$/$E_{true}$, $\sigma_{eff}$=%.2f'%eff_sig_raw)
    plt.legend(fontsize=13, loc='upper left')
    plt.xlabel('E/$E_{true}$',horizontalalignment='right', x=1.0, fontsize=14, labelpad=6)
    plt.ylabel('Events',horizontalalignment='right', y=1.0, fontsize=14, labelpad=6)
    #plt.show()
    plt.savefig('plots/'+plotname+'.pdf')
    plt.savefig('plots/'+plotname+'.png')


# In[7]:


plot_E_over_Etrue(Y_test_gamma1, Y_raw_test_gamma1, Y_predict_test_gamma1, "EoverEtrue_gamma1")


# ## Save to protobuf format

# In[13]:


loaded_model.input


# In[14]:


loaded_model.outputs


# In[15]:


loaded_model.output


# In[16]:


loaded_model.summary()


# In[7]:


import tensorflow as tf


# In[9]:


sess = tf.Session()
K.set_session(sess)


# In[10]:


K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)


# In[11]:


from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl


# In[22]:


prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={"input_0": loaded_model.input[0], "input_1": loaded_model.input[1]}, outputs={"prediction":loaded_model.output})


# In[23]:


builder = saved_model_builder.SavedModelBuilder('model_Pi0Net_CNN_dense_E_1000epoch.pb')


# In[24]:


legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


# In[25]:


init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# In[26]:


sess.run(init_op)


# In[27]:


builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)
# save the graph      
builder.save() 

