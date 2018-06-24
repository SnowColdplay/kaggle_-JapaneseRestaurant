import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from sklearn import *
#0.4981

#have done:
#1.ar,hr基本处理（reservation特征）
#2.距离，角度特征
#3.category one hot (genre,area)
#4.genre area cnt
#5.用dow做组合特征（genre,store,id,dow),hol组合特征
#6.概率特征
#问题6：holiday的表测一下线上，没用

#haven't done:
#问题1：hpg的信息没有包括进来，试了一下hpg加到air里，效果变差，不知为何，可能是hpg info里包含噪声？
#问题2：ar,hr的特征还可以再仔细考虑一下
#问题4：滑窗特征，当前时刻到之前的（gen,area,reservation数）,框架有了，理一下思路跑出来就行了
#问题5：黄金周的处理
#问题6：学习kkbox的svd
#问题7：学习seaborn的使用，自己会做统计图

print("Loading Data ...")
tra= pd.read_csv('air_visit_data.csv')
tes = pd.read_csv('sample_submission.csv')
print("done Loading Data ...")
print(tra.shape)
print(tes.shape)

#对ar,hr表做基本处理
def ar_hr(train,test):
    #已经把hr store id转化为了air id
    ar=pd.read_hdf('ar5.hdf')
    hr=pd.read_hdf('hr5.hdf')
    # for df in [ar,hr]:
    #     df['visit_date'] = pd.to_datetime(df['visit_date'])
    #     df['visit_date'] = df['visit_date'].dt.date
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
    asi=pd.read_hdf('asi_one_hot.hdf')
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

#按照 经纬度特征 进行count,(只有genre count的时候，才有用，area count没用）
#按照 经纬度特征 对visitoe进行处理，此处注意要加test_list = list(test['air_genre_name'].values)，不能把train加进去，否则过拟合
def geo_visitor(train,test):
    for col in ['air_genre_name']:
        data=train[['lon_plus_lat',col,'visit_date']].append(test[['lon_plus_lat',col,'visit_date']])
        tmp=data.groupby(['lon_plus_lat',col],as_index=False)['visit_date'].agg({str(col)+'_geo_cnt':'count'})
        train=pd.merge(train,tmp,on=['lon_plus_lat',col],how='left')
        test= pd.merge(test, tmp, on=['lon_plus_lat', col], how='left')

    train2 = train[(train['lon_plus_lat'].notnull()) & (train['air_genre_name'] != 0)]
    train2.groupby(['lon_plus_lat', 'air_genre_name'], as_index=False)['visitors'].agg({'visitors': [np.min, np.mean, np.median, np.max, np.size]})
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
        train[str(col) + '_cnt'] = train[col].apply(lambda x: ac[x])
        train[str(col) + '_cnt_log'] = np.log1p(train[str(col) + '_cnt'])
        test[str(col) + '_cnt'] = test[col].apply(lambda x: ac[x])
        test[str(col) + '_cnt_log'] = np.log1p(test[str(col) + '_cnt'])
    return train, test

#对store cnt
def store_cnt(train,test):
    #加log不管用
    data=train[['air_store_id','visit_date']].append(test[['air_store_id','visit_date']])
    store_cnt = data.groupby(by='air_store_id').count()['visit_date'].to_dict()
    train['store_cnt'] = train['air_store_id'].apply(lambda x: store_cnt[x])
    test['store_cnt'] = test['air_store_id'].apply(lambda x: store_cnt[x])
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
        #df['date_int'] = df['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
        df['year'] = df['visit_date'].dt.year
        df['month'] = df['visit_date'].dt.month
        df['dom'] = df['visit_date'].dt.day
        df['air_store_id2'] = lbl.fit_transform(df['air_store_id'])
    return train,test

#月映射特征
def map_month(m):
    m=int(m)
    if m<10:
        return 1
    elif m>=20:
        return 2
    else:
        return 3

def month_f(train,test):
    train['dom_map']=train['dom'].apply(map_month)
    test['dom_map'] = test['dom'].apply(map_month)
    for col in ['air_store_id','air_genre_name']:
        #加上area 是0.4959
        data = train[[col,'dom_map','visit_date']].append(test[[col, 'dom_map','visit_date']])
        dom_cnt = data.groupby(by=[col, 'dom_map'], as_index=False)['visit_date'].agg({str(col) + '_cnt_dom': 'count'})
        train = pd.merge(train, dom_cnt, on=[col, 'dom_map'], how='left')
        test = pd.merge(test, dom_cnt, on=[col, 'dom_map'], how='left')
    return train, test

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
    # hol['dom'] = hol['visit_date'].dt.day
    # hol['dom_map'] = hol['dom'].apply(map_month)
    # wkend = hol.apply(lambda x: (x.day_of_week == 'Sunday' or x.day_of_week == 'Saturday'), axis=1)
    # hol.loc[wkend, 'holiday_flg1'] = 1
    # 是否为上中下旬的节假日
    # yue_xun = [1, 2, 3]
    # col = ['xia_hol', 'zhong_hol', 'shang_hol']
    # for i in range(len(yue_xun)):
    #     hol[col[i]] = 0
    #     map_wkend = hol.apply(lambda x: (x.dom_map == yue_xun[i] or x.holiday_flg == 1), axis=1)
    #     hol.loc[map_wkend, col[i]] = 1
    #
    # del hol['dom'], hol['dom_map']
    #hol['visit_date'] = pd.to_datetime(hol['visit_date'])
    hol['day_of_week'] = lbl.fit_transform(hol['day_of_week'])
    train = pd.merge(train, hol, how='left', on=['visit_date'])
    test = pd.merge(test, hol, how='left', on=['visit_date'])
    return train,test

#按照store id 对visitor处理
def dow_visitor(train,test):
    #为什么unique stores加上train的之后变差了
    #加holiday flg不管用
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

#仿照kkbox第一名的时间戳
def timestamp(train,test):
    for col in ['air_store_id','air_genre_name','air_area_name']:
        data = train[[col, 'timestamp']].append(test[[col, 'timestamp']])
        data_mean = data.groupby(by=col).mean()['timestamp'].to_dict()
        train[str(col)+'_timestamp_mean'] = train [col].apply(lambda x: data_mean[x])
        test[str(col)+'_timestamp_mean'] = test[col].apply(lambda x: data_mean[x])

        data_std = data.groupby(by=col).std()['timestamp'].to_dict()
        train[str(col)+'_timestamp_std'] = train [col].apply(lambda x: data_std[x])
        test[str(col)+'_timestamp_std'] = test[col].apply(lambda x: data_std[x])

    return train, test

#按照是否holiday进行count
def cnt_hol(train,test):
    #加上周几，分数下降
    for col in ['air_store_id','air_genre_name','air_area_name']:
        data = train[[col,'holiday_flg','visit_date']].append(test[[col, 'holiday_flg','visit_date']])
        dow_cnt=data.groupby(by=[col,'holiday_flg'], as_index = False)['visit_date'].agg({str(col)+'_cnt_hol':'count'})
        train=pd.merge(train,dow_cnt,on=[col,'holiday_flg'],how='left')
        test= pd.merge(test, dow_cnt, on=[col,'holiday_flg'], how='left')
    #
    # for col in ['air_store_id']:
    #     for col_hol in ['xia_hol', 'zhong_hol', 'shang_hol']:
    #         data = train[[col,col_hol,'visit_date']].append(test[[col, col_hol,'visit_date']])
    #         dow_cnt=data.groupby(by=[col,col_hol], as_index = False)['visit_date'].agg({str(col)+str(col_hol)+'_cnt':'count'})
    #         train=pd.merge(train,dow_cnt,on=[col,col_hol],how='left')
    #         test= pd.merge(test, dow_cnt, on=[col,col_hol], how='left')
    return train, test


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

    train['store_dom_prob']=train['air_store_id_cnt_dom']/train['store_cnt']
    test['store_dom_prob'] = test['air_store_id_cnt_dom'] / test['store_cnt']

    train['genre_dom_prob'] = train['air_genre_name_cnt_dom'] / train['air_genre_name_cnt']
    test['genre_dom_prob'] = test['air_genre_name_cnt_dom'] / test['air_genre_name_cnt']

    # train['store_xiahol_prob'] = train['air_store_idxia_hol_cnt'] / train['store_cnt']
    # test['store_xiahol_prob'] = test['air_store_idxia_hol_cnt'] / test['store_cnt']
    #
    # train['store_zhonghol_prob'] = train['air_store_idzhong_hol_cnt'] / train['store_cnt']
    # test['store_zhonghol_prob'] = test['air_store_idzhong_hol_cnt'] / test['store_cnt']
    #
    # train['store_shanghol_prob'] = train['air_store_idshang_hol_cnt'] / train['store_cnt']
    # test['store_shanghol_prob'] = test['air_store_idshang_hol_cnt'] / test['store_cnt']

    del train['air_store_id_cnt_dow']
    del test['air_store_id_cnt_dow']

    return train, test

#时间窗口
def window(train,test):
    train1=pd.read_csv('window/train_win_store_cnt_online_before_10.csv')[['air_store_id','visit_date','store_cnt_before_10']]
    test1=pd.read_csv('window/test_win_store_cnt_online_before_10.csv')[['air_store_id','visit_date','store_cnt_before_10']]

    # train2=pd.read_csv('window/train_win_store_cnt_online_before_50.csv')[['air_store_id','visit_date','store_cnt_before_50']]
    # test2=pd.read_csv('window/test_win_store_cnt_online_before_50.csv')[['air_store_id','visit_date','store_cnt_before_50']]
    #
    # train3=pd.read_csv('window/train_win_store_cnt_online_after_10.csv')[['air_store_id','visit_date','store_cnt_after_10']]
    # test3=pd.read_csv('window/test_win_store_cnt_online_after_10.csv')[['air_store_id','visit_date','store_cnt_after_10']]
    for df in [train1,test1,train,test]:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_date'] = df['visit_date'].dt.date

    train=pd.merge(train,train1,on=['air_store_id','visit_date'],how='left')
    test = pd.merge(test,test1, on=['air_store_id','visit_date'], how='left')

    return train,test
#
# def pred_reserve(train,test):
#     train1=pd.read_csv('pred_reserve_xianshang_train.csv')
#     test1 = pd.read_csv('pred_reserve_xianshang_test.csv')
#     for df in [train1, test1, train, test]:
#         df['visit_date'] = pd.to_datetime(df['visit_date'])
#         df['visit_date'] = df['visit_date'].dt.date
#     train=pd.merge(train,train1,on=['air_store_id','visit_date'],how='left')
#     test = pd.merge(test,test1, on=['air_store_id','visit_date'], how='left')
#     return train,test

# def weather(train,test):
#     data=pd.read_hdf('data_weather.hdf')[['air_store_id','visit_date','precipitation','avg_temperature']]
#     for df in [train,test,data]:
#         df['visit_date'] = pd.to_datetime(df['visit_date'])
#     train=pd.merge(train,data,on=['air_store_id','visit_date'],how='left')
#     test = pd.merge(test, data, on=['air_store_id', 'visit_date'], how='left')
#     return train, test

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
    # train, test=golden_diff(train,test)
    # for df in [train, test]:
    #     df['visit_date'] = pd.to_datetime(df['visit_date'])
    #     df['visit_date'] = df['visit_date'].dt.date
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
    train, test = month_f(train, test)
    train, test=timestamp(train, test)
    train, test = prob(train, test)
    #train, test= window(train, test)
    #train, test = weather(train, test)
    #train, test = pred_reserve(train, test)
    print(train.shape)
    print(test.shape)
    print(train)
    train, test=fill_nan(train, test)
    return train,test

from sklearn import metrics
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

import lightgbm as lgb
def lgbCV(train,test):
    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
    X = train[col]
    y = np.log1p(train['visitors'].values)
    d_train = lgb.Dataset(X, y)
    watchlist_final = lgb.Dataset(X, y)
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
    lgb_model = lgb.train(params, train_set=d_train, num_boost_round=2000, valid_sets=watchlist_final, verbose_eval=5)
    predictors = [i for i in col]
    feat_imp = pd.Series(lgb_model.feature_importance(importance_type='split', iteration=-1), predictors).sort_values(ascending=False)
    feat_imp.to_csv('feature_importance.csv')
    test_probs = lgb_model.predict(test[col])
    #save
    test['visitors1']=test_probs
    test_save=test[['visitors1']]
    test_save.to_csv('lgb_pred.csv', index=False)
    del test['visitors1']
    #done save
    print('RMSE lgb ', RMSLE(np.log1p(test['visitors'].values),test_probs))


def sub(train,test):
    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
    X = train[col]
    y = np.log1p(train['visitors'].values)
    d_train = lgb.Dataset(X, y)
    watchlist_final = lgb.Dataset(X, y)
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
    params['seed'] = 2018
    lgb_model = lgb.train(params, train_set=d_train, num_boost_round=2000, valid_sets=watchlist_final, verbose_eval=5)
    predictors = [i for i in col]
    feat_imp = pd.Series(lgb_model.feature_importance(importance_type='split', iteration=-1), predictors).sort_values(
        ascending=False)
    feat_imp.to_csv('feature_importance_xianshang.csv')
    test_probs = lgb_model.predict(test[col])
    test_probs = np.expm1(test_probs)
    index_test = pd.read_csv('sample_submission.csv')
    index_test = index_test['id']
    result = pd.DataFrame({"id": index_test, "visitors": test_probs})
    result.to_csv('stacking/0124/LGB_sub_seed2018_onehot.csv', index=False)

if __name__ == "__main__":
    print('线下测试')
    train = tra[tra['visit_date'] < '2017-03-12']
    test = tra[tra['visit_date'] > '2017-03-12']
    for df in[train,test]:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_date'] = df['visit_date'].dt.date
    print("构造线下data的特征")
    train,test=make_feature(train,test)
    print("线下lgb")
    lgbCV(train, test)
    print('线上提交')
    tes['visit_date'] = tes['id'].map(lambda x: str(x).split('_')[2])
    tes['air_store_id'] = tes['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    train=tra
    test=tes
    for df in[train,test]:
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_date'] = df['visit_date'].dt.date
    print("构造线上data的特征")
    train,test=make_feature(train,test)
    # train.to_hdf('train0105.hdf', key='abc')
    # test.to_hdf('test0105.hdf', key='abc')
    print("线下lgb")
    sub(train, test)





