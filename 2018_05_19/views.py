# -*- coding: utf-8 -*-
from django.http import HttpResponse
from django.shortcuts import render
from __future__ import print_function
import mysql.connector
import json

from DataBase import DataBase  # query_data_set这个函数被加过一个参数
from algorithm import transforms, data_pca, kmeans_cluster
import numpy as np


# Create your views here.
def securitydrive(request):
    """
    APP发送一个car_id 算法用json返回用这个car_id之前的数据得到的危险点坐标 目前只有一个car_id的数据
    现在只能用app_data_store这个数据表里的long lat car_speed car_acc这些字段
    """
    # 连数据库
    db = DataBase()
    cur, connect = db.connect_db()

    # 数据库查询
    car_id = request.GET['car_id']
    query = 'SELECT long, lat, car_speed, car_acc FROM `app_data_store` WHERE car_id = %s'
    query_res = db.query_data_set(cur, query, [car_id])

    # !!! query_res需要转一下DataFrame格式 我不知道怎么转 !!!
    # 假装已经转好了

    # 算法部分 参考的你以前的代码
    location = np.array([query_res['long'].values, query_res['lat'].values]).astype("float64").transpose()
    data = query_res[['car_speed', 'car_acc']]
    data = transforms(data)

    feature_label = data_pca(data=data, stand=0.01)
    data = np.delete(data, feature_label, axis=1)
    labels = kmeans_cluster(data)

    abnormal_id = 0 if 2*np.sum(labels)>len(labels) else 1  # 两类中少的判为异常类

    res_data = {'coordinate': []}  # 返回json 格式: {'coordinate': [{'long': long, 'lat': lat}, ...]}

    for i in range(len(labels)):
        if labels[i] == abnormal_id:
            res_data['coordinate'].append({'long': location[i][0], 'lat': location[i][1]})

    return HttpResponse(json.dumps(res_data))


