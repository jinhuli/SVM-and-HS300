# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:22:02 2018

@author: 54326
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV as GSCV

class SVM():
    '''
    必须先建立类属性data，包含开高低收、交易量的日数据
    '''

    train_years=10 #训练样本的大小，十年数据
    year_days=240 #一年有240个交易日
    
    def __init__(self, momentum, span):
        self.m_span = momentum #动量的观测天数
        self.span = span #波动性的观测天数
        self.slippage = 0.0002 #滑点情况，共2个基点，买入卖出各1个基点
        
    def data_pre(self):
        '''
        数据准备，构建数据特征，并标记类别标签
        '''
        data = self.data
        data['retn'] = data['close'].pct_change()
        span = self.span
        
        attr = pd.DataFrame()
        #构建七个数据特征，刻画当天，周度和月度的市场状况
        attr['high_low'] = (data['high'] - data['low']) / data['open']
        attr['close_open'] = data['close'] / data['open'] - 1
        attr['vol_pct'] = data['volume'].pct_change()
        attr['pct_m'] = data['close'].shift(-self.m_span) / data['open'] - 1
        for i in range(span, len(data)-span):
            now = data.index[i]
            begin = data.index[i-span]
            attr.loc[now, 'high_v'] = data.loc[now, 'high']\
                                    / max(data.loc[begin:now, 'high']) - 1
            attr.loc[now, 'low_v'] = data.loc[now, 'low']\
                                   / min(data.loc[begin:now, 'low']) - 1
            attr.loc[now, 'sigma'] = data.loc[begin:now, 'retn'].std()\
                                   * self.year_days / span
        attr.dropna(inplace= True)
        
        #将特征标准化
        for field in attr.columns:
            attr[field] = (attr[field] - attr[field].mean()) / attr[field].std()
        plt.figure(figsize=(6, 5))
        sns.heatmap(attr.iloc[:, :-1].corr())
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('attr_heatmap.png')
        #上涨标记为1，下跌标记为-1
        attr['label'] = np.where(attr['close_open'].shift(-1) > 0, 1, -1)
        
        return attr
    
    def data_split(self, begin_year):
        '''
        数据划分，十年数据为训练样本，下一年的数据为测试样本
        Args:
            begin_year:
                第几年，0表示第一年，以此类推
        Returns:
            train_data:
                训练集
            test_data:
                测试集
        '''
        begin = begin_year * self.year_days #训练集开始的位置
        end = begin + self.train_years * self.year_days #训练集结束的位置
        attrs = self.data_pre()
        train_data = attrs.iloc[begin : end, :] #训练集
        test_data = attrs.iloc[end : end + self.year_days, :] #下一年数据为测试集
        return train_data, test_data
    
    def gscv_para(self, C_list, gamma_list, x_train, y_train):
        '''
        用网格搜索和交叉验证调节参数，考虑类别不平衡的情况,k-fold的k为10
        Args:
            C_list:
                C参数的备选列表
            gamma_list:
                gamma参数的备选列表
            x_train:
                训练集中的特征数据
            y_train:
                训练集中的类别数据
        Returns:
            最优的C参数，gamma参数和最优时的score(平均值)
            优化的标准是SVC的score值，score越高表示，表示参数越好
        '''
        clf = SVC(class_weight='balanced', cache_size=4000)
        gscv = GSCV(clf, param_grid={'C': C_list, 'gamma': gamma_list}, 
                    n_jobs=-1, cv=10, pre_dispatch=4)
        gscv.fit(x_train, y_train)
        return  gscv.best_params_.values(), gscv.best_score_
        
    def svm_fit(self, x_train, y_train):
        '''
        用网格搜索调参数
        '''
        C_list = list(range(1, 5))
        gamma_list = np.linspace(0.01, 0.05, 5)
        (C, gamma), score = self.gscv_para(C_list, gamma_list, x_train, y_train) 
        #可能存在参数在边界的情况，此时需要重置一下参数的备选范围
        if C == 1 or C == 10:
            if C == 1:
                print('C touched the lower limit')
                C_list = np.linspace(0.2, 2, 10)
            else:
                print('C touched the upper limit')
                C_list = list(range(1, 10))
            if gamma == 0.01 or gamma == 0.1:
                #C和gamma都在边界
                if gamma == 0.01:
                    print('gamma touched the lower limit')
                    gamma_list = np.linspace(0.002, 0.02, 10)
                else:
                    print('gamma touched the upper limit')
                    gamma_list = np.linspace(0.02, 0.2, 10)
            (C, gamma), score = self.gscv_para(C_list, gamma_list, x_train, y_train)
        else:
            pass
        
        if gamma == 0.01 or gamma == 0.1:
            #仅仅是gamma在边界
            if gamma == 0.01:
                print('gamma touched the lower limit')
                gamma_list = np.linspace(0.002, 0.02, 10)
            else:
                print('gamma touched the upper limit')
                gamma_list = np.linspace(0.02, 0.2, 10)
            (C, gamma), score = self.gscv_para(C_list, gamma_list, x_train, y_train)
        else:
            pass
        return C, gamma, score
    
    def predict(self, begin_year):
        '''
        用拟合的分类器对样本外的数据做预测
        '''
        train_data, test_data = self.data_split(begin_year)
        x_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        x_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
        C, gamma, score = self.svm_fit(x_train, y_train)
#        C, gamma, score, _ = paras[begin_year]
        clf = SVC(C=C, gamma=gamma, class_weight='balanced', cache_size=4000)
        clf.fit(x_train, y_train)
        test_score = clf.score(x_test, y_test) #测试集上的准确率
        para = C, gamma, score, test_score
        print('%d, %.4f, %.4f, %.4f' % para)
        #在测试集上的预测值
        y_predict = pd.Series(clf.predict(x_test), index=y_test.index, name='predict')
        return para, pd.concat([y_test, y_predict], axis=1)
        
    def cum_retn(self, years):
        retns = {}
        paras = []
        for i in range(years):
            para, label = self.predict(i)
            paras.append(para)
            for j in range(len(label)):
                date = label.index[j]
                if label.loc[date, 'label'] == label.loc[date, 'predict']:
                    #预测正确，则获取多空收益
                    retn_i = abs(data.loc[date, 'close'] / data.loc[date, 'open'] - 1)
                    retns[date] = retn_i - self.slippage
                else:
                    #预测失误，产生损失
                    retn_i = -abs(data.loc[date, 'close'] / data.loc[date, 'open'] - 1)
                    retns[date] = retn_i - self.slippage
        return paras, 1 + pd.Series(retns).sort_index().cumsum()
        
if __name__ == '__main__':   
    
    data = pd.read_excel('E:/Data/HS300_05_18.xlsx', index_col='date')
    SVM.data = data
    hs = SVM(5, 20)
    attr = hs.data_pre()

    years = int(len(data) / 240) + 1
    paras, nav = hs.cum_retn(years - hs.train_years)
    def drawdown(nav):
        #计算最大回撤
        dd = []
        for i in range(1, len(nav)):
            max_i = max(nav[:i])
            dd.append(min(0, nav[i] - max_i) / max_i)
        return dd
    Drawdown = pd.Series(drawdown(nav), index=nav.index[1:])
    maxdd = min(Drawdown) 
    calmar = - (nav[-1]**(1/4) -1) / maxdd 
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(nav, label='strategy')
    hs300 = data['close'][-len(nav):] / data['close'][-len(nav)]
    ax1.set_xlim(nav.index[0], nav.index[-1])
    ax1.plot(hs300, label='HS300')
    ax1.set_ylabel('Net Asset Value', fontdict={'fontsize':16})
    ax1.set_xlabel('Date', fontdict={'fontsize':16})
    ax1.legend(loc='lower right', fontsize=14)
    ax2 = ax1.twinx()
    ax2.set_ylim(-1.5, 0)
    ax2.plot(Drawdown, color='c')
    ax2.set_ylabel('Max Drawdown', fontdict={'fontsize':16})
    ax2.fill_between(Drawdown.index, Drawdown, color='c')
    ax2.set_ylim(-1.5, 0)
    plt.savefig('v_svm.png', bbox_inches='tight')
    
    for i in range(4):
        train_data, test_data = hs.data_split(i)
        x_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        x_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
        C, gamma, *res = paras[i]
        clf = SVC(C=C, gamma=gamma, class_weight='balanced', cache_size=4000)
        clf.fit(x_train, y_train)
        print('%.4f' % clf.score(x_train, y_train))
    