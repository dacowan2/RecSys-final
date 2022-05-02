import numpy as np
import pandas as pd
import os
import warnings
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['figure.figsize'] = [12,8]
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=15.5)    # legend fontsize
plt.rc('figure', titlesize=50)  # fontsize of the figure title

header = ['user_id','item_id','rating','timestamp']
dataset = pd.read_csv('../data/ml-100k/u.data',sep = '\t',names = header)

n_items = len(pd.unique(dataset['item_id']))
n_users = len(pd.unique(dataset['user_id']))

train, test_temp = train_test_split(dataset, test_size=0.2, random_state=42)
val, test = train_test_split(test_temp, test_size=0.5, random_state=42)

model_num = 80
n_factors_list = [5,25,50,100,200]
n_nodes_per_layer_list = [64, 32, 16, 8, 4, 2]
lr_list = [1e-1, 1e-2, 1e-3, 1e-4]
dropout_prob = 0.2
epochs = 250
batch_size = 256
patience = 5
early_stopping_metric = 'val_loss'

for i, n_factors in enumerate(n_factors_list):
    for j, lr in enumerate(lr_list):
        
        print('model num: ')
        print(model_num)
        
        parent_dir = 'models/'
        path = os.path.join(parent_dir, f'model_{model_num}')
        try:
            os.mkdir(path)
        except OSError as error:
            print('There is already a folder for this model. Try another model number.')

        parent_dir = f'models/model_{model_num}'
        path = os.path.join(parent_dir, 'figures')
        try:
            os.mkdir(path)
        except OSError as error:
            print('There is already a folder for this model. Try another model number.')

        # creating item embedding path
        movie_input = Input(shape=[1], name="Item-Input")
        movie_embedding = Embedding(n_items+1, n_factors, name="Item-Embedding")(movie_input)
        movie_vec = Flatten(name="Flatten-Items")(movie_embedding)

        # creating user embedding path
        user_input = Input(shape=[1], name="User-Input")
        user_embedding = Embedding(n_users+1, n_factors, name="User-Embedding")(user_input)
        user_vec = Flatten(name="Flatten-Users")(user_embedding)

        # concatenate features
        conc = Concatenate()([movie_vec, user_vec])

        # add fully-connected-layers
        dense = Dense(n_nodes_per_layer_list[0], activation='relu')(conc)
        dropout = Dropout(dropout_prob)(dense)
        batch_norm = BatchNormalization()(dropout)

        for k, n_nodes in enumerate(n_nodes_per_layer_list[1:-1]):
            dense = Dense(n_nodes, activation='relu')(batch_norm)
            dropout = Dropout(dropout_prob)(dense)
            batch_norm = BatchNormalization()(dropout)

        dense = Dense(n_nodes_per_layer_list[-1], activation='relu')(batch_norm)
        out = Dense(1)(dense)

        # Create model and compile it
        model = Model([user_input, movie_input], out)
        model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())
        
        callback = EarlyStopping(monitor=early_stopping_metric, patience=patience)

        history = model.fit(x = [train.user_id, train.item_id], y = train.rating, validation_data = ((val.user_id, val.item_id), val.rating), epochs=epochs, verbose=1, batch_size = batch_size, callbacks = [callback])

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(train_loss, label = 'train')
        plt.plot(val_loss, label = 'val')
        plt.yscale('log')
        plt.ylabel('mse loss')
        plt.xlabel('epochs')
        plt.title(f'Model {model_num}: Loss Curves')
        plt.legend()
        plt.savefig(f'models/model_{model_num}/figures/loss.png')
        plt.close()

        predictions = model.predict([test.user_id, test.item_id])
        preds_std = np.std(predictions)

        predictions_list = []
        for pred_rating in predictions:
            predictions_list.append(pred_rating[0])

        ratings_preds_array = np.array(predictions_list).astype('float64')
        ratings_actual_array = np.array(test.rating)

        test_mse = mean_squared_error(ratings_actual_array, ratings_preds_array)

        # make csv with ensemble info
        model_info_header_list = ['model', 'test mse', 'test preds std', 'epochs', 'learning rate', 'n_nodes_per_layer', 'n_factors', 'batch_size', 'dropout_prob', 'patience', 'early stopping metric']
        model_info_val_list = [[model_num, test_mse, preds_std, epochs, lr, n_nodes_per_layer_list, n_factors, batch_size, dropout_prob, patience, early_stopping_metric]]

        model_info_df = pd.DataFrame(model_info_val_list, columns = model_info_header_list)
        model_info_df.to_csv(f'models/model_{model_num}/model_info.csv')
        
        model_num += 1
        
        print('--------------------------------------------------------------------------------------')