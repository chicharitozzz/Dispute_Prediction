# @Time : 2018/8/27 9:15 
# @Author : Chicharito_Ron
# @File : arma_pred.py 
# @Software: PyCharm Community Edition
# 时间序列预测，ARIMA模型

import numpy as np
import pandas as pd
import cx_Oracle as oracle
import os
import json
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'  # 数据库编码设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ArimaPrediction:
    def __init__(self, db_url, areacode, disputetype):
        self.db_url = db_url  # 数据库地址
        self.areacode = areacode
        self.distype = disputetype

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

    def getts(self, datas):
        """根据区号、纠纷类型筛选时序序列"""
        year_mon = datas[:, :2].astype(str)  # 组合年月
        date = []
        for d in year_mon:
            date.append('-'.join(d))

        # 生成数据框
        df = pd.DataFrame(datas[:, 2:], columns=['区号', '纠纷类型', '数量'], index=date)

        # 根据区号、纠纷类型筛选数据
        ts = df['数量'][(df['区号'] == self.areacode) & (df['纠纷类型'] == self.distype)].astype(float)['2016-1':'2017-12']

        return ts

    @staticmethod
    def data_visualization(ts):
        """数据可视化: 绘制自相关系数、偏自相关系数图"""
        fig, axes = plt.subplots(3, 1)

        ts.plot(marker='*', markersize=5, label='纠纷数量', ax=axes[0])
        axes[0].set_title('纠纷数量走势图')
        axes[0].legend()

        plot_pacf(ts, lags=18, ax=axes[1])  # acf图
        plot_acf(ts, lags=18, ax=axes[2])  # pacf图

        plt.subplots_adjust(hspace=1)
        plt.show()

    @staticmethod
    def stationary_processing(ts):
        """"时序数据平稳性处理:默认进行一阶差分"""
        # roll_ts = ts.rolling(window=12, min_periods=8).mean().dropna()  # 滑动平均
        diff_ts = ts.diff(1).dropna()  # 一阶差分

        return diff_ts

    @staticmethod
    def stationarity_test(ts):
        """时序数据平稳性检测:Dickey Fuller检验"""
        dftest = adfuller(ts)

        dfres = pd.Series(dftest[:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

        for key, value in dftest[4].items():
            dfres['Critical Value {}'.format(key)] = value

        print(dfres)

        return dfres

    @staticmethod
    def time_random_test(ts):
        """时间序列随机性检测:LjungBox
        p值低于0.05则拒绝原假设，原序列为非随机序列"""
        pvalue = acorr_ljungbox(ts, lags=1)[1][0]
        print('pvalue=', pvalue)

    @staticmethod
    def para_selection(ts):
        """ARIMA模型参数选择:p、q参数"""
        pmax = qmax = len(ts) // 5  # 参数范围

        bic_li = []  # bic矩阵
        for p in range(pmax + 1):
            tmp = []
            for q in range(qmax + 1):
                try:
                    tmp.append(ARIMA(ts, order=(p, 1, q)).fit().bic)
                except:
                    tmp.append(None)
            bic_li.append(tmp)

        print(bic_li)

        # 选出使bic最小的p值和q值
        bic = pd.DataFrame(bic_li)
        p, q = bic.stack().idxmin()
        return p, q

    @staticmethod
    def arima_pred(ts, p, q, n):
        """模型建立和预测
        n:预测的月份数"""
        model = ARIMA(ts, (p, 1, q)).fit()
        res = model.forecast(n)[0].astype(int).tolist()  # 预测后面几个月的数量

        for i in range(len(res)):
            if res[i] < 0:
                res[i] = 0

        pi = pd.period_range(ts.index[-1], periods=n+1, freq='M')[1:]  # 生成预测的月份
        pred_ts = pd.Series(res, index=pi)
        nts = ts.append(pred_ts)

        # 绘制预测走势图
        nts.plot()
        ts.plot()
        plt.title('纠纷数量预测走势图')
        plt.show()

        print(pred_ts)
        return res


if __name__ == '__main__':
    with open('./static/815数据.json', encoding='utf-8') as f:
        datas = np.array(json.load(f))

    dbbase = 'pkdata1/pkdata1@192.168.0.13:1521/orcl'
    ap = ArimaPrediction(dbbase, 320111, 99)
    # datas = ap.read_db()
    ts = ap.getts(datas)

    # 预处理
    # ap.data_visualization(ts)
    # diff_ts = ap.stationary_processing(ts)
    # ap.stationarity_test(diff_ts)
    # ap.time_random_test(diff_ts)

    # 模型拟合
    p, q = ap.para_selection(ts)
    ap.arima_pred(ts, p, q, 4)
    print(p, q)
