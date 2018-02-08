import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings 环境变量

df, promo_df, items, stores = load_unstack('all')

# data after 2015
df = df[pd.date_range(date(2015,6,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2015,6,1), date(2017,8,31))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

timesteps = 200

# preparing data
#注意这里的生成器
train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 7, 5),
                                           n_range=16, day_skip=7, batch_size=2000, aux_as_tensor=False, reshape_output=2)
Xval, Yval = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 7, 26),
                                     aux_as_tensor=False, reshape_output=2)
Xtest, _ = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 8, 16),
                                    aux_as_tensor=False, is_train=False, reshape_output=2)

w = (Xval[7][:, 2] * 0.25 + 1) / (Xval[7][:, 2] * 0.25 + 1).mean() # validation weight: 1.25 if perishable and 1 otherwise per competition rules

del df, promo_df; gc.collect()

print('current no promo 2') # log info

#卷积核的数目
latent_dim = 32

# Define input
# seq input
#1维长度为timesteps的向量
#model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], output)
seq_in = Input(shape=(timesteps, 1))
is0_in = Input(shape=(timesteps, 1))
promo_in = Input(shape=(timesteps+16, 1))
yearAgo_in = Input(shape=(timesteps+16, 1))
quarterAgo_in = Input(shape=(timesteps+16, 1))
item_mean_in = Input(shape=(timesteps, 1))
store_mean_in = Input(shape=(timesteps, 1))
# store_family_mean_in = Input(shape=(timesteps, 1))

weekday_in = Input(shape=(timesteps+16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=timesteps+16)(weekday_in) #7,4为inputdim：字典长度和outputdim：全连接嵌入的维度
# weekday_embed_decode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
dom_in = Input(shape=(timesteps+16,), dtype='uint8')
dom_embed_encode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# dom_embed_decode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# weekday_onehot = Lambda(K.one_hot, arguments={'num_classes': 7}, output_shape=(timesteps+16, 7))(weekday_in)

# aux input
cat_features = Input(shape=(6,))
item_family = Lambda(lambda x: x[:, 0, None])(cat_features)
item_class = Lambda(lambda x: x[:, 1, None])(cat_features)
item_perish = Lambda(lambda x: x[:, 2, None])(cat_features)
store_nbr = Lambda(lambda x: x[:, 3, None])(cat_features)
store_cluster = Lambda(lambda x: x[:, 4, None])(cat_features)
store_type = Lambda(lambda x: x[:, 5, None])(cat_features)

#为啥只embedding了其中4个
# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=1)(item_family)
# class_embed = Embedding(337, 8, input_length=1)(item_class)
store_embed = Embedding(54, 8, input_length=1)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=1)(store_cluster)
type_embed = Embedding(5, 2, input_length=1)(store_type)

encode_slice = Lambda(lambda x: x[:, :timesteps, :])
# encode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in], axis=2)
# encode_features = encode_slice(encode_features)

x_in = concatenate([seq_in, encode_slice(promo_in), item_mean_in], axis=2)

# Define network
# c0 = TimeDistributed(Dense(4))(x_in)
# # c0 = Conv1D(4, 1, activation='relu')(sequence_in)
'''
kernel_size==2,strides步长默认1，dilation_rate扩张率：间隔卷积
“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。
“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同
'''
c1 = Conv1D(latent_dim, 2, dilation_rate=1, padding='causal', activation='relu')(x_in)
c2 = Conv1D(latent_dim, 2, dilation_rate=2, padding='causal', activation='relu')(c1)
c2 = Conv1D(latent_dim, 2, dilation_rate=4, padding='causal', activation='relu')(c2)
c2 = Conv1D(latent_dim, 2, dilation_rate=8, padding='causal', activation='relu')(c2)
# c2 = Conv1D(latent_dim, 2, dilation_rate=16, padding='causal', activation='relu')(c2)

c4 = concatenate([c1, c2])
# c2 = MaxPooling1D()(c2)

conv_out = Conv1D(8, 1, activation='relu')(c4)
# conv_out = GlobalAveragePooling1D()(c4)
conv_out = Dropout(0.25)(conv_out)
conv_out = Flatten()(conv_out)

decode_slice = Lambda(lambda x: x[:, timesteps:, :])
promo_pred = decode_slice(promo_in)
# qAgo_pred = decode_slice(quarterAgo_in)
# yAgo_pred = decode_slice(yearAgo_in)


# Raw sequence in results overfitting!!!
dnn_out = Dense(512, activation='relu')(Flatten()(seq_in))
dnn_out = Dense(256, activation='relu')(dnn_out)
# dnn_out = BatchNormalization()(dnn_out)
dnn_out = Dropout(0.25)(dnn_out)

x = concatenate([conv_out, dnn_out,
                 Flatten()(promo_pred), Flatten()(family_embed), Flatten()(store_embed), Flatten()(cluster_embed), Flatten()(type_embed), item_perish])
# x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
# x = Dense(256, activation='relu')(x)
# x = BatchNormalization()(x)
# x = concatenate([x, seq_in])
output = Dense(16, activation='relu')(x)

model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], output)

# rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_data, steps_per_epoch=1000, workers=4, use_multiprocessing=True, epochs=10, verbose=2,
                    validation_data=(Xval, Yval, w))
'''
当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
batch_size=2000
verbose,2为每个epoch输出一行记录
'''

val_pred = model.predict(Xval)
cal_score(Yval, val_pred)

test_pred = model.predict(Xtest)
make_submission(df_index, test_pred, 'cnn_no-promo2.csv')
# gc.collect()

# model.save('save_models/cnn_model')