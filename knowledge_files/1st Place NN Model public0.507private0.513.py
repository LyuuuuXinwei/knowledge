"""
This is an upgraded version of Ceshine's and Linzhi and Andy Harless starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")

stores = pd.read_csv(
    "../input/stores.csv",
).set_index("store_nbr")

le = LabelEncoder()
items['family'] = le.fit_transform(items['family'].values)

stores['city'] = le.fit_transform(stores['city'].values)
stores['state'] = le.fit_transform(stores['state'].values)
stores['type'] = le.fit_transform(stores['type'].values)

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False) #test独有且打折？？？被省略了
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)  #fill0
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))
stores = stores.reindex(df_2017.index.get_level_values(0))


df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()  #商品的每日全国销量
promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()  #商品每日全国打折店铺数

df_2017_store_class = df_2017.reset_index()
df_2017_store_class['class'] = items['class'].values
df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]
df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()  #每日同商品大类class&store的总商品销量

df_2017_promo_store_class = promo_2017.reset_index()
df_2017_promo_store_class['class'] = items['class'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['class', 'store_nbr']]
df_2017_promo_store_class = df_2017_promo_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()  #每日同商品大类class&store的总打折数

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(df, promo_df, t2017, is_train=True, name_prefix=None):
    X = {
        #14,60,140天内总打折次数
        "promo_14_2017": get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_df, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_df, t2017, 140, 140).sum(axis=1).values,
        #分割点后3,7,14天的累积打折次数
        "promo_3_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values,
        "promo_7_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,
        "promo_14_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,
    }
    #分割点前连续视窗
    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values #平均差分
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values #np.arange(i)[::-1]倒置，衰减加权累加
        X['mean_%s' % i] = tmp.mean(axis=1).values #平均
        X['median_%s' % i] = tmp.median(axis=1).values #中位数
        X['min_%s' % i] = tmp.min(axis=1).values #最小值
        X['max_%s' % i] = tmp.max(axis=1).values #最大
        X['std_%s' % i] = tmp.std(axis=1).values #标准差

    #分割点一周前作为新的分割点，新分割点前的连续视窗的上述统计值
    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017 + timedelta(days=-7), i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    #分割点前的连续视窗
    for i in [7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values # 非零记录数
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values #最后有销量的日子距今
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values #最早有记录的日子距今

        tmp = get_timespan(promo_df, t2017, i, i)
        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values #打折累积次数
        X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values #最后打折日距今
        X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values #最早打折日距今

    #分割点后一天-16天
    tmp = get_timespan(promo_df, t2017 + timedelta(days=16), 15, 15)
    X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values #累计打折次数
    X['last_has_promo_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values #测试窗最后打折日
    X['first_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values #测试窗最早打折日

    #分割点前16天的每日数据
    for i in range(1, 16):
        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()

    #分割点28天-22天的未来4周X平均和xxx20周周X平均
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    #分割点的16天前至16天后的每日打折情况
    for i in range(-16, 16):
        X["promo_{}".format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns] #构造列名前缀
    return X

print("Preparing dataset...")
num_days = 8
t2017 = date(2017, 5, 31) #后8周作为训练集
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(df_2017, promo_2017, t2017 + delta)

    #df_2017_item是商品group的每日总销量表
    X_tmp2 = prepare_dataset(df_2017_item, promo_2017_item, t2017 + delta, is_train=False, name_prefix='item')
    X_tmp2.index = df_2017_item.index
    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

    #df_2017_store_class是store class聚合总销量表
    X_tmp3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, t2017 + delta, is_train=False, name_prefix='store_class')
    X_tmp3.index = df_2017_store_class.index
    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)

    #所有的特征为三表分别得到并并入+items和store的label encoder编码
    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)

    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

del X_l, y_l
#验证集日期：7.26后16日
X_val, y_val = prepare_dataset(df_2017, promo_2017, date(2017, 7, 26))
X_val2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 7, 26), is_train=False, name_prefix='item')
X_val2.index = df_2017_item.index
X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_val3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')
X_val3.index = df_2017_store_class.index
X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)

X_test = prepare_dataset(df_2017, promo_2017, date(2017, 8, 16), is_train=False)
X_test2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 8, 16), is_train=False, name_prefix='item')
X_test2.index = df_2017_item.index
X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_test3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 8, 16), is_train=False, name_prefix='store_class')
X_test3.index = df_2017_store_class.index
X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_test = pd.concat([X_test, X_test2, X_test3, items.reset_index(), stores.reset_index()], axis=1)
del df_2017_item, promo_2017_item, df_2017_store_class, df_2017_promo_store_class, df_2017_store_class_index
gc.collect()

scaler = StandardScaler()
scaler.fit(pd.concat([X_train, X_val, X_test]))
#所有数据包括item也标准化？
X_train[:] = scaler.transform(X_train)
X_val[:] = scaler.transform(X_val)
X_test[:] = scaler.transform(X_test)

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
X_val = X_val.as_matrix()
#reshape？？？
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))


def build_model():
    #1层LSTM+7层全连接，递减dropout（防止过拟合），每层BatchNormalization，参数化relu
    model = Sequential()
    model.add(LSTM(512, input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(.2))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(64))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(16))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(1))

    return model

N_EPOCHS = 2000

val_pred = []
test_pred = []
# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.
sample_weights=np.array( pd.concat([items["perishable"]] * num_days) * 0.25 + 1 )

#每天单独预测
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    y = y_train[:, i]
    y_mean = y.mean()
    xv = X_val
    yv = y_val[:, i]
    model = build_model()
    opt = optimizers.Adam(lr=0.001) #adam优化
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    #回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        #当评价指标不在提升时，减少学习率
        #epsilon：阈值，用来确定是否进入检测值的“平原区”
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        ]
    #verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    model.fit(X_train, y - y_mean, batch_size = 65536, epochs = N_EPOCHS, verbose=2,
               sample_weight=sample_weights, validation_data=(xv,yv-y_mean), callbacks=callbacks ) #为什么要用-y_mean？
    val_pred.append(model.predict(X_val)+y_mean)
    test_pred.append(model.predict(X_test)+y_mean)

weight = items["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).squeeze(axis=2).transpose())**2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / 16)
print('nwrmsle = {}'.format(err))

y_val = np.array(val_pred).squeeze(axis=2).transpose() #squeeze，把size是1的捏在一起，axis越大越内层
df_preds = pd.DataFrame(
    y_val, index=df_2017.index,
    columns=pd.date_range("2017-07-26", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
df_preds["unit_sales"] = np.clip(np.expm1(df_preds["unit_sales"]), 0, 1000) #clip最大值最小值限定
df_preds.reset_index().to_csv('nn_cv.csv', index=False)

print("Making submission...")
y_test = np.array(test_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('nn_sub.csv', float_format='%.4f', index=None)