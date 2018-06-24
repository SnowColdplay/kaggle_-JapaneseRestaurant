import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from sklearn import *
import xgboost
import lightgbm as lgb

print("Loading Data ...")
tra= pd.read_csv('air_visit_data.csv')
tes = pd.read_csv('sample_submission.csv')
print("done Loading Data ...")
print(tra.shape)
print(tes.shape)

def ar_hr(train,test):
    ar=pd.read_hdf('ar5.hdf')
    hr=pd.read_hdf('hr5.hdf')
    for df in [ar,hr]:
        train=pd.merge(train,df, how='left', on=['air_store_id','visit_date'])
        test = pd.merge(test, df, how='left', on=['air_store_id', 'visit_date'])
    return train,test

def deal_ar_hr(train,test):
    del_col = ['Hour', 'sum', 'mean', 'diff']
    for j in range(len(del_col)):
        ar_col = [c for c in train if '_ar' in c and del_col[j] in c]
        hr_col = [c for c in train if '_hr' in c and del_col[j] in c]
        mean_col = []  # 把末尾的ar和hr去掉
        for i in range(len(ar_col)):
            mean_col.append(ar_col[i][:-3])
        for i in range(len(ar_col)):
            train[mean_col[i]] = (train[ar_col[i]] + train[hr_col[i]]) / 2
            test[mean_col[i]] = (test[ar_col[i]] + test[hr_col[i]]) / 2
    return train, test

#对col的类别进行one hot编码，以及本身做编码处理
def genre_area(train,test):
    asi=pd.read_hdf('asi_no_one_hot.hdf')
    lbl = preprocessing.LabelEncoder()
    for col in ['air_genre_name','air_area_name','air_area_name0','air_genre_name0','air_genre_name1','area_gen','area0_gen']:
        asi[col] = lbl.fit_transform(asi[col])
    train = pd.merge(train, asi, how='left', on=['air_store_id'])
    test = pd.merge(test, asi, how='left', on=['air_store_id'])
    return train,test

#经纬度一些特征（surprise me里的）
def geo(train,test):
    train['var_max_lat'] = train['latitude'].max() - train['latitude']
    train['var_max_long'] = train['longitude'].max() - train['longitude']
    test['var_max_lat'] = test['latitude'].max() - test['latitude']
    test['var_max_long'] = test['longitude'].max() - test['longitude']
    train['lon_plus_lat'] = train['longitude'] + train['latitude']
    test['lon_plus_lat'] = test['longitude'] + test['latitude']
    return train, test

def geo_visitor(train,test):
    for col in ['air_genre_name']:
        data=train[['lon_plus_lat',col,'visit_date']].append(test[['lon_plus_lat',col,'visit_date']])
        tmp=data.groupby(['lon_plus_lat',col],as_index=False)['visit_date'].agg({str(col)+'_geo_cnt':'count'})
        train=pd.merge(train,tmp,on=['lon_plus_lat',col],how='left')
        test= pd.merge(test, tmp, on=['lon_plus_lat', col], how='left')

    train2 = train[(train['lon_plus_lat'].notnull()) & (train['air_genre_name'] != 0)]
    tmp = train2.groupby(['lon_plus_lat', 'air_genre_name'], as_index=False)['visitors'].agg({'visitors': [np.min, np.mean, np.median, np.max, np.size]})
    tmp.columns = ['lon_plus_lat', 'air_genre_name', 'min_visitors1', 'mean_visitors1', 'median_visitors1', 'max_visitors1','count_observations1']
    test_list = list(test['air_genre_name'].values)
    tmp = tmp[tmp.air_genre_name.isin(test_list)]
    train=pd.merge(train,tmp,on=['lon_plus_lat','air_genre_name'],how='left')
    test= pd.merge(test, tmp, on=['lon_plus_lat', 'air_genre_name'], how='left')
    return train, test

#按照store, genre,area进行cnt
def area_genre_cnt(train,test):
    #不能去重，去重效果变差。
    for col in ['air_genre_name', 'air_area_name', 'air_genre_name0',
                'air_area_name0','area_gen','area0_gen']:
        data = train[['air_store_id', col]].append(test[['air_store_id', col]])
        ac = data.groupby(by=col).count()['air_store_id'].to_dict()
        train[str(col) + '_cnt'] = train[col].apply(lambda x: ac[x] if x in ac else -1)
        train[str(col) + '_cnt_log'] = np.log1p(train[str(col) + '_cnt'])
        test[str(col) + '_cnt'] = test[col].apply(lambda x: ac[x] if x in ac else -1)
        test[str(col) + '_cnt_log'] = np.log1p(test[str(col) + '_cnt'])
    return train, test

#对store cnt
def store_cnt(train,test):
    #加log不管用
    data=train[['air_store_id','visit_date']].append(test[['air_store_id','visit_date']])
    store_cnt = data.groupby(by='air_store_id').count()['visit_date'].to_dict()
    train['store_cnt'] = train['air_store_id'].apply(lambda x: store_cnt[x] if x in store_cnt else -1)
    test['store_cnt'] = test['air_store_id'].apply(lambda x: store_cnt[x] if x in store_cnt else -1)
    return train, test

#周几对应的store,area,genre count
def dow_cnt(train,test):
    #area按照每周count下降
    for col in ['air_store_id','air_genre_name','air_area_name']:
        data = train[[col, 'dow','visit_date']].append(test[[col, 'dow','visit_date']])
        dow_cnt=data.groupby(by=[col,'dow'], as_index = False)['visit_date'].agg({str(col)+'_cnt_dow':'count'})
        train=pd.merge(train,dow_cnt,on=[col,'dow'],how='left')
        test= pd.merge(test, dow_cnt, on=[col, 'dow'], how='left')
    return train, test

def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L
def cal_degree(lon1,lat1,lon2,lat2):
    dx = (lon2 - lon1) * np.cos(lat1 / 57.2958)
    dy = lat2-lat1
    degree = 180-90*(np.copysign(1,dy))-np.arctan(dx/dy)*57.2958
    degree = round(degree/45)*45
    return degree

#对经纬度聚类，得到中心点，以及到中心点的距离，角度。
#到东西南北点，边界的距离和角度 不管用
def geo_f(train,test):
    col = ['air_store_id', 'visit_date', 'latitude', 'longitude']
    data = train[col].append(test[col])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data[['longitude', 'latitude']])
    data['cluster'] = kmeans.predict(data[['longitude', 'latitude']])
    center_lon = [c[0] for c in kmeans.cluster_centers_]
    center_lat = [c[1] for c in kmeans.cluster_centers_]
    c = pd.DataFrame({'center_lon': center_lon, 'center_lat': center_lat, 'cluster': range(10)})
    data = pd.merge(data, c, on='cluster', how='left')
    data['center_dis'] = cal_distance(data['latitude'], data['longitude'], data['center_lat'], data['center_lon'])
    data['center_dis_log']=data['center_dis'].apply(lambda x: math.log(x))
    data['center_angle'] = cal_degree(data['latitude'], data['longitude'], data['center_lat'], data['center_lon'])
    # 最近点最远点
    for col in ['latitude', 'longitude']:
        tmp = data.groupby(['cluster'], as_index=False)[col].max().rename(columns={col: 'max_' + str(col)[:3]})
        data = pd.merge(data, tmp, on='cluster', how='left')
        tmp = data.groupby(['cluster'], as_index=False)[col].min().rename(columns={col: 'min_' + str(col)[:3]})
        data = pd.merge(data, tmp, on='cluster', how='left')
    data['max_lat_diff'] = data['max_lat'] - data['latitude']
    data['max_long_diff'] = data['max_lon'] - data['longitude']
    del data['cluster']
    train = pd.merge(train, data, on=['air_store_id', 'visit_date', 'latitude', 'longitude'], how='left')
    test = pd.merge(test, data, on=['air_store_id', 'visit_date', 'latitude', 'longitude'], how='left')
    return train, test


def other(train,test):
    lbl = preprocessing.LabelEncoder()
    for df in[train,test]:
        df['year'] = df['visit_date'].dt.year
        df['month'] = df['visit_date'].dt.month
        #df['air_store_id'] = df['air_store_id'].astype(object)
        df['air_store_id2'] = lbl.fit_transform(df['air_store_id'].values)
    return train,test

#加入timestimp
def holiday(train,test):
    train['visit_date'] = pd.to_datetime(train['visit_date'])
    test['visit_date'] = pd.to_datetime(test['visit_date'])
    lbl = preprocessing.LabelEncoder()
    hol = pd.read_csv('date_info.csv')
    hol = hol.rename(columns={'calendar_date': 'visit_date'})
    #加入timestamp
    hol['timestamp'] = range(len(hol))
    hol['timestamp'] = hol['timestamp'].map(lambda x: x * 24 * 3600)
    hol['visit_date'] = pd.to_datetime(hol['visit_date'])
    hol['day_of_week'] = lbl.fit_transform(hol['day_of_week'])
    train = pd.merge(train, hol, how='left', on=['visit_date'])
    test = pd.merge(test, hol, how='left', on=['visit_date'])
    return train, test

def dow_visitor(train,test):
    train['visit_date'] = pd.to_datetime(train['visit_date'])
    train['dow'] = train['visit_date'].dt.dayofweek
    test['visit_date'] = pd.to_datetime(test['visit_date'])
    test['dow'] = test['visit_date'].dt.dayofweek
    tmp = train.groupby(['air_store_id', 'dow'], as_index=False).agg({'visitors': [np.min, np.mean, np.median, np.max, np.size]})
    tmp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors','count_observations']
    test_store_list=list(test['air_store_id'].values)
    tmp=tmp[tmp.air_store_id.isin (test_store_list)]
    train = pd.merge(train, tmp, how='left', on=['air_store_id', 'dow'])
    test = pd.merge(test, tmp, how='left', on=['air_store_id', 'dow'])
    return train,test

#概率
def prob(train,test):
    train['store_dow_prob']=train['air_store_id_cnt_dow']/train['store_cnt']
    test['store_dow_prob'] = test['air_store_id_cnt_dow'] / test['store_cnt']

    train['genre_dow_prob']=train['air_genre_name_cnt_dow']/train['air_genre_name_cnt']
    test['genre_dow_prob'] = test['air_genre_name_cnt_dow'] / test['air_genre_name_cnt']

    train['area_dow_prob']=train['air_area_name_cnt_dow']/train['air_area_name_cnt']
    test['area_dow_prob'] = test['air_area_name_cnt_dow'] / test['air_area_name_cnt']

    train['store_hol_prob'] = train['air_store_id_cnt_hol'] / train['store_cnt']
    test['store_hol_prob'] = test['air_store_id_cnt_hol'] / test['store_cnt']

    train['genre_hol_prob'] = train['air_genre_name_cnt_hol'] / train['air_genre_name_cnt']
    test['genre_hol_prob'] = test['air_genre_name_cnt_hol'] / test['air_genre_name_cnt']

    train['area_hol_prob'] = train['air_area_name_cnt_hol'] / train['air_area_name_cnt']
    test['area_hol_prob'] = test['air_area_name_cnt_hol'] / test['air_area_name_cnt']

    del train['air_store_id_cnt_dow']
    del test['air_store_id_cnt_dow']

    return train, test

#仿照kkbox第一名的时间戳
def timestamp(train,test):
    for col in ['air_store_id','air_genre_name','air_area_name']:
        data = train[[col, 'timestamp']].append(test[[col, 'timestamp']])
        data_mean = data.groupby(by=col).mean()['timestamp'].to_dict()
        train[str(col)+'_timestamp_mean'] = train [col].apply(lambda x: data_mean[x] if x in data_mean else -1)
        test[str(col)+'_timestamp_mean'] = test[col].apply(lambda x: data_mean[x] if x in data_mean else -1)
        data_std = data.groupby(by=col).std()['timestamp'].to_dict()
        train[str(col)+'_timestamp_std'] = train [col].apply(lambda x: data_std[x] if x in data_std else -1)
        test[str(col)+'_timestamp_std'] = test[col].apply(lambda x: data_std[x] if x in data_std else -1)
    return train, test

#按照是否holiday进行count
def cnt_hol(train,test):
    #加上周几，分数下降
    for col in ['air_store_id','air_genre_name','air_area_name']:
        data = train[[col,'holiday_flg','visit_date']].append(test[[col, 'holiday_flg','visit_date']])
        dow_cnt=data.groupby(by=[col,'holiday_flg'], as_index = False)['visit_date'].agg({str(col)+'_cnt_hol':'count'})
        train=pd.merge(train,dow_cnt,on=[col,'holiday_flg'],how='left')
        test= pd.merge(test, dow_cnt, on=[col,'holiday_flg'], how='left')
    return train, test
#时间窗口
def window(train,test):
    train1=pd.read_csv('window/train_win_store_cnt_online_before_10.csv')[['air_store_id','visit_date','store_cnt_before_10']]
    test1=pd.read_csv('window/test_win_store_cnt_online_before_10.csv')[['air_store_id','visit_date','store_cnt_before_10']]
    for df in [train1,test1,train,test]:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_date'] = df['visit_date'].dt.date
    train=pd.merge(train,train1,on=['air_store_id','visit_date'],how='left')
    test = pd.merge(test,test1, on=['air_store_id','visit_date'], how='left')
    return train,test

import math
def golden_map(data):
    #2016年
    if str(data).split('-')[0]=='2016':
        if str(data)>='2016-04-29' and str(data)<= '2016-05-05':
            return 0
        elif str(data)<'2016-04-29':
            data0= pd.to_datetime('2016-04-29')
            data = pd.to_datetime(data)
            #向右取整
            return math.ceil((data0-data).days/7)
        elif str(data)>'2016-05-05':
            data0= pd.to_datetime('2016-05-05')
            data = pd.to_datetime(data)
            #向右取整
            return math.ceil((data-data0).days/7)
    if str(data).split('-')[0] == '2017':
        if str(data)>='2017-04-29' and str(data)<='2017-05-05':
            return 0
        elif str(data)<'2017-04-29':
            data0= pd.to_datetime('2017-04-29')
            data = pd.to_datetime(data)
            #向右取整
            return math.ceil((data0-data).days/7)
        elif str(data)>'2017-05-05':
            data0= pd.to_datetime('2017-05-05')
            data = pd.to_datetime(data)
            #向右取整
            return math.ceil((data-data0).days/7)

def golden_diff(train,test):
    train['golden_map'] = train['visit_date'].apply(golden_map)
    test['golden_map'] = test['visit_date'].apply(golden_map)
    #把黄金周的visitor用前一周和后一周的均值填充，dow
    train['visit_date'] = pd.to_datetime(train['visit_date'])
    train['dow'] = train['visit_date'].dt.dayofweek
    test['visit_date'] = pd.to_datetime(test['visit_date'])
    test['dow'] = test['visit_date'].dt.dayofweek

    tmp=train[train['golden_map']==1] #相差一周的
    tmp1=tmp.groupby(['air_store_id', 'dow'], as_index=False).agg({'visitors': [np.mean]})
    tmp1.columns = ['air_store_id', 'dow', 'mean_visitors_gm']
    train=pd.merge(train,tmp1,on=['air_store_id', 'dow'],how='left')

    tmp2=tmp.groupby(['air_store_id'], as_index=False).agg({'visitors': [np.mean]})
    tmp2.columns = ['air_store_id', 'mean_visitors_gm2']
    train = pd.merge(train, tmp2, on=['air_store_id'], how='left')

    tmp=test[test['golden_map']==1] #相差一周的
    tmp1=tmp.groupby(['air_store_id', 'dow'], as_index=False).agg({'visitors': [np.mean]})
    tmp1.columns = ['air_store_id', 'dow', 'mean_visitors_gm']
    test=pd.merge(test,tmp1,on=['air_store_id', 'dow'],how='left')

    tmp2=tmp.groupby(['air_store_id'], as_index=False).agg({'visitors': [np.mean]})
    tmp2.columns = ['air_store_id', 'mean_visitors_gm2']
    test = pd.merge(test, tmp2, on=['air_store_id'], how='left')

    train.loc[train['golden_map']==0, 'visitors'] = train.loc[train['golden_map']==0, 'mean_visitors_gm']
    test.loc[test['golden_map'] == 0, 'visitors'] = test.loc[test['golden_map'] == 0, 'mean_visitors_gm']

    train.loc[train.visitors.isnull(), 'visitors'] = train.loc[train.visitors.isnull(), 'mean_visitors_gm2']
    test.loc[test.visitors.isnull(), 'visitors'] = test.loc[test.visitors.isnull(), 'mean_visitors_gm2']
    del train['golden_map'],train['dow'],train['mean_visitors_gm'],train['mean_visitors_gm2']
    del test['golden_map'], test['dow'], test['mean_visitors_gm'],test['mean_visitors_gm2']
    return train,test

def fill_nan(train,test):
    train = train.fillna(-1)
    test = test.fillna(-1)
    return train, test

def make_feature(train,test):
    train,test=ar_hr(train,test)
    train, test=deal_ar_hr(train, test)
    train, test= genre_area(train, test)
    train, test=area_genre_cnt(train, test)
    train, test= geo(train,test)
    train, test=geo_visitor(train, test)
    train, test = geo_f(train, test)
    train, test= store_cnt(train, test)
    train, test=holiday(train, test)
    train, test=cnt_hol(train, test)
    train, test = dow_visitor(train, test)
    train, test =dow_cnt(train, test)
    train, test = other(train, test)
    train, test=timestamp(train, test)
    #train, test = prob(train, test)
    #train, test = window(train, test)
    train, test=fill_nan(train, test)
    print(train.shape)
    print(test.shape)
    return train,test

#########################################开始构造模型#########################################
from sklearn import metrics
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

def blending_offline(train,test):
    print('lgb')
    print('load data')
    y_train= np.log1p(train['visitors'].values)  #blending用

    dataset_blend_train = np.zeros((train.shape[0], 2))
    dataset_blend_test = np.zeros((test.shape[0], 2))
    dataset_blend_test_j = np.zeros((test.shape[0], 5))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    lgb_pred= np.zeros(shape=[len(test)])
    i = -1
    for train_indices, val_indices in kf.split(train):
        i = i + 1

        train_origi=train.copy()
        train,val=make_feature(train.loc[train_indices, :], train.loc[val_indices, :])  #训练
        train_all, test= make_feature(train_origi, test)   #验证
        col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
        print(train)
        print(val)

        train_data = lgb.Dataset(train[col],label= np.log1p(train['visitors']))
        val_data = lgb.Dataset(val[col],label= np.log1p(val['visitors']))

        print('Training LGBM model...')
        params = {}
        params['application'] = 'regression'
        params['boosting'] = 'gbdt'
        params['learning_rate'] = 0.01
        params['num_leaves'] = 200
        params['max_depth'] = 20
        params['min_sum_hessian_in_leaf'] = 1e-2
        params['min_gain_to_split'] = 0
        params['bagging_fraction'] = 0.5
        params['feature_fraction'] = 0.5
        params['metric'] = 'rmse'
        lgb_model = lgb.train(params, train_set=train_data, num_boost_round=2000, valid_sets=[val_data],verbose_eval=5,early_stopping_rounds=25)

        y_submission = lgb_model.predict(train[col])  # 每折的预测
        dataset_blend_train[val_indices, 0] = y_submission
        dataset_blend_test_j[:, i] = lgb_model.predict(test[col])
        #每折的结果
        lgb_pre_each = lgb_model.predict(test[col])
        print('每折的 lgb RMSE', RMSLE(np.log1p(test['visitors'].values),  lgb_pre_each ))
        #五折取平均
        lgb_pred+=lgb_model.predict(test[col])
        del lgb_model
    print('5折平均lgb RMSE', RMSLE(np.log1p(test['visitors'].values),  lgb_pred/5))
    dataset_blend_test[:,0] = dataset_blend_test_j.mean(1)

    print('xgb')
    dataset_blend_test_j = np.zeros((test.shape[0], 5))
    kf = KFold(n_splits=5)
    xgb_pred = np.zeros(shape=[len(test)])
    i = -1
    for train_indices, val_indices in kf.split(train):
        i = i + 1
        col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
        d_train=  xgboost.DMatrix(train[col].loc[train_indices, :], label=np.log1p(train.loc[train_indices, 'visitors']))
        d_valid =  xgboost.DMatrix(train[col].loc[val_indices, :], label=np.log1p(train.loc[val_indices, 'visitors']))
        d_test = xgboost.DMatrix(test[col])

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        params = {
            'learning_rate': 0.05,
            'objective': 'reg:linear',
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'n_estimators': 200,
            'max_depth': 8,
            'seed': 1,
            'eval_metric': 'rmse'

        }
        model = xgboost.train(params, d_train, 642, watchlist, verbose_eval=10)
        y_submission = model.predict(d_valid)  # 每折的预测
        dataset_blend_train[val_indices, 1] = y_submission
        dataset_blend_test_j[:, i] = model.predict(d_test)
        #每折的结果
        xgb_pred_each= model.predict(d_test)
        print('每折的 xgb RMSE', RMSLE(np.log1p(test['visitors'].values),xgb_pred_each))
        # 五折取平均
        xgb_pred += model.predict(d_test)
        del model
    print('5折平均lgb RMSE', RMSLE(np.log1p(test['visitors'].values),  xgb_pred/5))
    dataset_blend_test[:, 1] = dataset_blend_test_j.mean(1)

    print("Blending.")
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()

    # 模型预测结果(几个模型就有几列），真实结果（1列）
    clf.fit(dataset_blend_train, y_train)
    from sklearn.externals import joblib
    # 储存模型
    joblib.dump(clf, "LR.m")
    print('w0',clf.coef_)
    print('w1', clf.intercept_)
    print(dataset_blend_train)
    print(y_train)
    print(dataset_blend_test)
    # 模型对test的预测结果（几个模型，test就有几列）
    y_submission = clf.predict(dataset_blend_test)
    print(y_submission)
    print('blending RMSE', RMSLE(np.log1p(test['visitors'].values), y_submission))


def blending_online(train,test):
    print('lgb')
    print('load data')
    y_train= np.log1p(train['visitors'].values)  #blending用

    dataset_blend_train = np.zeros((train.shape[0], 2))
    dataset_blend_test = np.zeros((test.shape[0], 2))
    dataset_blend_test_j = np.zeros((test.shape[0], 5))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    lgb_pred= np.zeros(shape=[len(test)])
    i = -1
    for train_indices, val_indices in kf.split(train):
        i = i + 1
        col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
        train_data = lgb.Dataset(train[col].loc[train_indices, :],label= np.log1p(train.loc[train_indices, 'visitors']))
        val_data = lgb.Dataset(train[col].loc[val_indices, :],label= np.log1p(train.loc[val_indices, 'visitors']))

        print('Training LGBM model...')
        params = {}
        params['application'] = 'regression'
        params['boosting'] = 'gbdt'
        params['learning_rate'] = 0.01
        params['num_leaves'] = 300
        params['max_depth'] = 25
        params['min_sum_hessian_in_leaf'] = 1e-2
        params['min_gain_to_split'] = 0
        params['bagging_fraction'] = 0.5
        params['feature_fraction'] = 0.5
        params['metric'] = 'rmse'
        lgb_model = lgb.train(params, train_set=train_data, num_boost_round=2000, valid_sets=[val_data],verbose_eval=5,early_stopping_rounds=25)

        y_submission = lgb_model.predict(train[col].loc[val_indices, :])  # 每折的预测
        dataset_blend_train[val_indices, 0] = y_submission
        dataset_blend_test_j[:, i] = lgb_model.predict(test[col])
        del lgb_model
    dataset_blend_test[:,0] = dataset_blend_test_j.mean(1)

    print('xgb')
    dataset_blend_test_j = np.zeros((test.shape[0], 5))
    kf = KFold(n_splits=5)
    xgb_pred = np.zeros(shape=[len(test)])
    i = -1
    for train_indices, val_indices in kf.split(train):
        i = i + 1
        col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
        d_train=  xgboost.DMatrix(train[col].loc[train_indices, :], label=np.log1p(train.loc[train_indices, 'visitors']))
        d_valid =  xgboost.DMatrix(train[col].loc[val_indices, :], label=np.log1p(train.loc[val_indices, 'visitors']))
        d_test = xgboost.DMatrix(test[col])

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        params = {
            'learning_rate': 0.05,
            'objective': 'reg:linear',
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'n_estimators': 200,
            'max_depth': 8,
            'seed': 1,
            'eval_metric': 'rmse'

        }
        model = xgboost.train(params, d_train, 642, watchlist, verbose_eval=10,early_stopping_rounds=25)
        y_submission = model.predict(d_valid)  # 每折的预测
        dataset_blend_train[val_indices, 1] = y_submission
        dataset_blend_test_j[:, i] = model.predict(d_test)
        del model
    dataset_blend_test[:, 1] = dataset_blend_test_j.mean(1)

    print("Blending.")
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()

    # 模型预测结果(几个模型就有几列），真实结果（1列）
    clf.fit(dataset_blend_train, y_train)
    from sklearn.externals import joblib
    # 储存模型
    joblib.dump(clf, "LR.m")
    print('w0',clf.coef_)
    print('w1', clf.intercept_)
    print(dataset_blend_train)
    print(y_train)
    print(dataset_blend_test)
    # 模型对test的预测结果（几个模型，test就有几列）
    y_submission = clf.predict(dataset_blend_test)
    print(y_submission)
    test_probs=y_submission
    test_probs = np.expm1(test_probs)
    index_test = pd.read_csv('sample_submission.csv')
    index_test = index_test['id']
    result = pd.DataFrame({"id": index_test, "visitors": test_probs})
    result.to_csv('blending_sub_0107.csv', index=False)



if __name__ == "__main__":
    print("Loading Data ...")
    tra = pd.read_csv('air_visit_data.csv')
    tes = pd.read_csv('sample_submission.csv')
    print("done Loading Data ...")
    print(tra.shape)
    print(tes.shape)

    print('线下测试')
    train = tra[tra['visit_date'] < '2017-03-12']
    test = tra[tra['visit_date'] > '2017-03-12']

    for df in [train, test]:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_date'] = df['visit_date'].dt.date
    print('平滑黄金周')
    train, test = golden_diff(train, test)
    for df in [train, test]:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_date'] = df['visit_date'].dt.date
    print('done')

    print("构造线下data的特征")
    #train,test=make_feature(train,test)
    blending_offline(train,test)

    # print('线上提交')
    # tes['visit_date'] = tes['id'].map(lambda x: str(x).split('_')[2])
    # tes['air_store_id'] = tes['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    # train = tra
    # test = tes
    # for df in [train, test]:
    #     df['visit_date'] = pd.to_datetime(df['visit_date'])
    #     df['visit_date'] = df['visit_date'].dt.date
    # print("构造线上data的特征")
    # train, test = make_feature(train, test)
    # print("线下lgb")
    # blending_online(train, test)