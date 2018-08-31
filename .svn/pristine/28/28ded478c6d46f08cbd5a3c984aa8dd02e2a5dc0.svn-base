# @Time : 2018/8/20 9:02
# @Author : Chicharito_Ron
# @File : predapi.py
# @Software: PyCharm Community Edition
# 对数据库中数据进行训练并建立模型，对下个月的纠纷数量进行预测，将结果写入库中。

import numpy as np
import pandas as pd
import json
import cx_Oracle as oracle
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor


os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'  # 数据库编码设置


class DisputePrediction:
    def __init__(self, db_url):
        self.db_url = db_url  # 数据库地址

    def read_db(self):
        """读数据库"""
        db = oracle.connect(self.db_url)
        cur = db.cursor()
        sql = 'SELECT YEAR, MONTH, AREACODE, TYPE, REALCOUNT FROM RMTJ_PREDICT WHERE REALCOUNT IS NOT NULL'
        cur.execute(sql)
        datas = cur.fetchall()
        cur.close()
        db.close()
        datas = np.array(datas, dtype=int)
        # print(datas.shape[0])

        return datas

    @staticmethod
    def reset(datas):
        """获取区域信息、矛盾类型、月份信息"""
        y_o, m_o = datas[-1, :2]
        if m_o == 12:
            y_n = y_o + 1
            m_n = 1
        else:
            y_n = y_o
            m_n = m_o + 1

        # print(y_n, m_n)
        areacodes = list(set(datas[:, 2]))  # 区号
        types = list(set(datas[:, 3]))  # 纠纷类型

        return y_n, m_n, areacodes, types

    @staticmethod
    def training(datas):
        """模型训练与存储"""
        # 生成数据框
        # df = pd.DataFrame(data=datas, columns=['年', '月', '区号', '纠纷类型', '数量'])

        # 区号数值化
        # le = LabelEncoder()
        # le.fit(df['区号'])
        # joblib.dump(le, './static/le.pkl')  # 模型存储
        # areacode = le.transform(df['区号'])

        # 数组横向合并
        # con_data = np.concatenate((datas[:, 1].reshape(-1, 1), areacode.reshape(-1, 1), datas[:, 3:]), axis=1)

        # 数据与标签生成
        # X = con_data[:, :3]
        # Y = con_data[:, -1]
        X = datas[:, :4]  # 特征中加入年份
        Y = datas[:, -1]

        # 特征one-hot编码
        ohe = OneHotEncoder(n_values='auto', sparse=False)
        ohe.fit(X)

        joblib.dump(ohe, './static/ohe.pkl')  # 模型存储
        X_vec = ohe.transform(X)

        # 回归模型初始化
        rf = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=8, random_state=100, n_jobs=-1)
        rf.fit(X_vec, Y)

        joblib.dump(rf, './static/rf.pkl')

        return

    def predict_next_month(self, y_n, m_n, areacodes, types):
        """预测下个月的数据,
        将预测结果存入数据库中
        单个样本格式: [年，月，区号，纠纷类型]"""
        # 载入模型
        ohe = joblib.load('./static/ohe.pkl')
        rf = joblib.load('./static/rf.pkl')

        # 生成样本集
        samples = []
        for area in areacodes:
            for t in types:
                samples.append([y_n, m_n, area, t])

        # 生成特征向量数组
        f_vec = ohe.transform(np.array(samples))

        # 预测
        rf_pred = np.around(rf.predict(f_vec)).astype(int)

        # 预测数据写入数据库
        res_li = np.concatenate((samples, rf_pred.reshape(-1, 1)), axis=1).tolist()
        db = oracle.connect(self.db_url)
        cur = db.cursor()

        for r in res_li:
            # sql = 'INSERT INTO RMTJ_PREDICT (YEAR, MONTH, AREACODE, TYPE, RF_PRED) VALUES{}'.format(tuple(r))

            sql = 'UPDATE RMTJ_PREDICT SET RF_PRED={0} WHERE YEAR={1} AND MONTH={2} AND AREACODE={3} AND TYPE={4}'\
                .format(r[4], r[0], r[1], r[2], r[3])

            cur.execute(sql)
            print('插入', r)

        db.commit()
        cur.close()
        db.close()

        return rf_pred

    def main(self):
        # with open('./static/815数据.json', encoding='utf-8') as f:
        #     datas = np.array(json.load(f))

        datas = self.read_db()
        y_n, m_n, areacodes, types = self.reset(datas)
        self.training(datas)
        rf_pred = self.predict_next_month(y_n, m_n, areacodes, types)

        return rf_pred

    @staticmethod
    def predict(sample):
        """对新样本进行预测，
        样本格式: [年，月，区号，纠纷类型]
        """
        le = joblib.load('./static/le.pkl')
        ohe = joblib.load('./static/ohe.pkl')
        rf = joblib.load('./static/rf.pkl')

        # 区号转化
        ac = le.transform([sample[2]])
        sample[2] = ac[0]

        # 特征向量生成
        x_vec = ohe.transform(np.array(sample[1:]).reshape(1, -1))[0]

        # 预测
        pre = round(rf.predict(x_vec.reshape(1, -1))[0])

        print('预测结果为:', pre)

        return pre


if __name__ == '__main__':
    # dbbase = 'pkdata1/pkdata1@192.168.0.13:1521/orcl'
    dbbase = sys.argv[1]
    dp = DisputePrediction(dbbase)
    pres = dp.main()
    print(pres[:10])
