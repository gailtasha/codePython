# Projeto Marinha do Brasil

# Autor: Natanael Junior (natmourajr@gmail.com)
# Laboratorio de Processamento de Sinais - UFRJ

import os
import pickle
import numpy as np
import time

from sklearn.decomposition import PCA
from sklearn.externals import joblib

init_time = time.time()

m_time = time.time()
print 'Time to import all libraries: '+str(m_time-init_time)+' seconds'

outputpath = os.environ['OUTPUTDATAPATH']
main_analysis_path = os.environ['SONAR_WORKSPACE']
log_analysis_path = os.environ['PACKAGE_OUTPUT']
result_analysis_path = os.environ['PACKAGE_OUTPUT']+'/PCDDeflaction'
pict_results_path = os.environ['PACKAGE_OUTPUT']+'/PCDDeflaction/picts'
files_results_path = os.environ['PACKAGE_OUTPUT']+'/PCDDeflaction/output_files'

# Read data
# Check if LofarData has created...
m_time = time.time()


subfolder = '4classes'
n_pts_fft = 1024
decimation_rate = 3

if(not os.path.exists(outputpath+'/'+'LofarData_%s_%i_fft_pts_%i_decimation_rate.jbl'%(
            subfolder,n_pts_fft,decimation_rate))):
    print outputpath+'/'+'LofarData_%s_%i_fft_pts_%i_decimation_rate.jbl'%(
        subfolder,n_pts_fft,decimation_rate)+' doesnt exist...please create it'
    exit()
    
#Read lofar data
[data,class_labels] = joblib.load(outputpath+'/'+
                                  'LofarData_%s_%i_fft_pts_%i_decimation_rate.jbl'%(
            subfolder,n_pts_fft,decimation_rate))
m_time = time.time()-m_time
print 'Time to read data file: '+str(m_time)+' seconds'

# Get data in correct format
from keras.utils import np_utils

# create a full data vector
all_data = {};
all_trgt = {};

for iclass, class_label in enumerate(class_labels):
    for irun in range(len(data[iclass])):
        if len(all_data) == 0:
            all_data = data[iclass][irun]['Signal']
            all_trgt = (iclass)*np.ones(data[iclass][irun]['Signal'].shape[1])
        else:
            all_data = np.append(all_data,data[iclass][irun]['Signal'],axis=1)
            all_trgt = np.append(all_trgt,(iclass)*np.ones(data[iclass][irun]
                                                           ['Signal'].shape[1]),axis=0)

all_data = all_data.transpose()

# turn targets in sparse mode
trgt_sparse = np_utils.to_categorical(all_trgt)

# Train Process
from Functions import LogFunctions as log

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.layers import Merge

# Create a entry in log file
m_log = log.LogInformation()
date = m_log.CreateLogEntry(package_name="PreProcessing",analysis_name='PCDDeflaction')

# Create a train information file
n_folds = 2
n_inits = 1
n_pcds = 3
norm = 'mapstd'

train_info = {}
train_info['n_folds'] = n_folds
train_info['n_inits'] = n_inits
train_info['n_pcds'] = n_pcds
train_info['norm'] = norm

train_info_name = result_analysis_path+'/train_info_files'+'/'+date+'_train_info.jbl'
classifiers_name = result_analysis_path+'/classifiers_files'+'/'+date+'_classifiers'
pdf_file_name = result_analysis_path+'/output_files'+'/'+date+'_pcds'

from sklearn import cross_validation
from sklearn import preprocessing

CVO = cross_validation.StratifiedKFold(all_trgt, train_info['n_folds'])
CVO = list(CVO)
train_info['CVO'] = CVO

joblib.dump([train_info],train_info_name,compress=9)

# train classifiers
classifiers = {}
trn_desc = {}
pcds = {}

# try to estimate time to be done...
total_trains = train_info['n_folds']*train_info['n_inits']
nn_trained = 0

for ifold in range(train_info['n_folds']):
    train_id, test_id = CVO[ifold]
    
    # normalize data based in train set
    if train_info['norm'] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(all_data[train_id,:])
    elif train_info['norm'] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(all_data[train_id,:])
    elif train_info['norm'] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(all_data[train_id,:])
        
    norm_all_data = scaler.transform(all_data)
       
    classifiers[ifold] = {}
    trn_desc[ifold] = {}
    pcds[ifold] = {}
    
    for ipcd in range(train_info['n_pcds']):
        best_init = 0
        best_loss = 999
        if ipcd == 0:
            # first pcd: regular NN
            for i_init in range(train_info['n_inits']):
                print ('Fold: %i of %i - PCD: %i of %i - Init: %i of %i'
                       %(ifold+1, train_info['n_folds'],
                         ipcd+1, train_info['n_pcds'],
                         i_init+1,train_info['n_inits']))
                model = Sequential()
                model.add(Dense(all_data.shape[1],
                                input_dim=all_data.shape[1], 
                                init='identity',trainable=False))
                model.add(Activation('linear'))
                model.add(Dense(1, input_dim=all_data.shape[1], init='uniform'))
                model.add(Activation('tanh'))
                model.add(Dense(trgt_sparse.shape[1], init='uniform')) 
                model.add(Activation('tanh'))
                
                sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss='mean_squared_error', optimizer=sgd
                      ,metrics=['accuracy'])
                
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=25, 
                                            verbose=0, mode='auto')
                # Train model
                init_trn_desc = model.fit(norm_all_data[train_id], trgt_sparse[train_id], 
                                nb_epoch=50, 
                                batch_size=8,
                                callbacks=[earlyStopping], 
                                verbose=0,
                                validation_data=(norm_all_data[test_id],
                                                 trgt_sparse[test_id]),
                                shuffle=True)
                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = i_init
                    best_loss = np.min(init_trn_desc.history['val_loss'])
                    classifiers[ifold][ipcd] = model
                    trn_desc[ifold][ipcd] = init_trn_desc
                    pcd_test = model.layers[2].get_weights()
                    pcds[ifold][ipcd] = pcd_test[0]
                
        else:
            # from second to end: freeze previous and train only last one 
            for i_init in range(train_info['n_inits']):
                print ('Fold: %i of %i - PCD: %i of %i - Init: %i of %i'
                       %(ifold+1, train_info['n_folds'],
                         ipcd+1, train_info['n_pcds'],
                         i_init+1,train_info['n_inits']))
                model = Sequential()
                
                #freezing the first layer
                freeze_layer = []
                trn_data = []
                tst_data = []
                
                for jpcd in range(ipcd):
                    buffer_layer = Sequential()
                    buffer_layer.add(Dense(1, 
                                           input_dim=norm_all_data.shape[1],
                                           trainable=False))
                    w = buffer_layer.get_weights()
                    w[0] = pcds[ifold][jpcd]
                    buffer_layer.set_weights(w)
                    
                    if jpcd == 0:
                        freeze_layer = buffer_layer
                        trn_data = norm_all_data[train_id]
                        tst_data = norm_all_data[test_id]
                    else:
                        freeze_layer = Merge([freeze_layer, non_freeze_layer],
                                             mode='concat')
                        if jpcd == 1:
                            trn_data = [trn_data, norm_all_data[train_id]]
                            tst_data = [tst_data, norm_all_data[test_id]]
                        else:
                            trn_data = [trn_data[0], norm_all_data[train_id]]
                            tst_data = [tst_data[0], norm_all_data[test_id]]
                    
                no_freeze_layer = Sequential()
                no_freeze_layer.add(Dense(1,
                                          input_dim=norm_all_data.shape[1],
                                          trainable=True))
                if ipcd == 1:
                    trn_data= [trn_data, norm_all_data[train_id]]
                    tst_data= [tst_data, norm_all_data[test_id]]
                else:
                    trn_data= [trn_data[0], norm_all_data[train_id]]
                    tst_data= [tst_data[0], norm_all_data[test_id]]
                    
                first_layer = Merge([freeze_layer, no_freeze_layer], mode='concat')
                model.add(first_layer)
                model.add(Activation('tanh'))
                model.add(Dense(trgt_sparse.shape[1], init='uniform')) 
                model.add(Activation('tanh'))
                
                
                sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss='mean_squared_error',
                              optimizer=sgd, 
                              metrics=['accuracy'])
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=25,
                                                        verbose=0,
                                                        mode='auto')
                
                # Train model
                init_trn_desc = model.fit(trn_data, trgt_sparse[train_id], 
                                nb_epoch=50, 
                                batch_size=8,
                                callbacks=[earlyStopping], 
                                verbose=0,
                                validation_data=(tst_data,trgt_sparse[test_id]),
                                shuffle=True)
                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = i_init
                    best_loss = np.min(init_trn_desc.history['val_loss'])
                    classifiers[ifold][ipcd] = model
                    trn_desc[ifold][ipcd] = init_trn_desc
                    
                    # in keras, we cannot access directly the weights
                    # one layer for weights and one for activation
                    pcds[ifold][ipcd] = model.get_weights()[2*ipcd]
#classifiers_file = open(classifiers_name+'.pickle', "wb")
#pickle.dump([classifiers,trn_desc],classifiers_file)
#classifiers_file.close()

pdfs_file = open(pdf_file_name+'.pickle', "wb")
pickle.dump([pcds],pdfs_file)
pdfs_file.close()

tst_data

# Train Process
from Functions import LogFunctions as log

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.layers import Merge

# Create a entry in log file
m_log = log.LogInformation()
date = m_log.CreateLogEntry(package_name="PreProcessing",analysis_name='PCDDeflaction')

# Create a train information file
n_folds = 2
n_inits = 1
n_pcds = 2
norm = 'mapstd'

train_info = {}
train_info['n_folds'] = n_folds
train_info['n_inits'] = n_inits
train_info['n_pcds'] = n_pcds
train_info['norm'] = norm

train_info_name = result_analysis_path+'/train_info_files'+'/'+date+'_train_info.jbl'
classifiers_name = result_analysis_path+'/classifiers_files'+'/'+date+'_classifiers'
pdf_file_name = result_analysis_path+'/output_files'+'/'+date+'_pcds'

from sklearn import cross_validation
from sklearn import preprocessing

CVO = cross_validation.StratifiedKFold(all_trgt, train_info['n_folds'])
CVO = list(CVO)
train_info['CVO'] = CVO

joblib.dump([train_info],train_info_name,compress=9)

# train classifiers
classifiers = {}
trn_desc = {}
pcds = {}

# try to estimate time to be done...
total_trains = train_info['n_folds']*train_info['n_inits']
nn_trained = 0

for ifold in range(train_info['n_folds']):
    train_id, test_id = CVO[ifold]
    
    # normalize data based in train set
    if train_info['norm'] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(all_data[train_id,:])
    elif train_info['norm'] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(all_data[train_id,:])
    elif train_info['norm'] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(all_data[train_id,:])
        
    norm_all_data = scaler.transform(all_data)
       
    print 'Train Process for %i Fold of %i Folds'%(ifold+1,train_info['n_folds'] )
    classifiers[ifold] = {}
    trn_desc[ifold] = {}
    pcds[ifold] = {}
    
    model = Sequential()
    
    freeze_layer = Sequential()
    freeze_layer.add(Dense(1, input_dim=norm_all_data.shape[1]))

    non_freeze_layer = Sequential()
    non_freeze_layer.add(Dense(1, input_dim=norm_all_data.shape[1]))
    
    merged = Merge([freeze_layer, non_freeze_layer], mode='concat')
    model.add(merged)
    model.add(Activation('tanh'))
    model.add(Dense(trgt_sparse.shape[1], init='uniform')) 
    model.add(Activation('tanh'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, 
                  metrics=['accuracy'])
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=25,
                                            verbose=0, 
                                            mode='auto')
    # Train model
    
    train_data = [norm_all_data[train_id], norm_all_data[train_id]]
    val_data = [norm_all_data[test_id], norm_all_data[test_id]]
    
    init_trn_desc = model.fit(train_data, 
                              trgt_sparse[train_id],
                              nb_epoch=50, 
                              batch_size=8, 
                              callbacks=[earlyStopping], 
                              verbose=0, 
                              validation_data=(val_data,
                                               trgt_sparse[test_id]),
                              shuffle=True)
    break

model.get_weights()



