# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 23:24:02 2018

@author: 54326
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV as GSCV

class SVM():
    
#    data = data #开高低收、成交量的日数据
    train_years=10 #训练样本的大小，十年数据
    year_days=240 #一年有240个交易日
    
    def __init__(self, backward, forward):
        self.bk = backward #回看前几天的数据
        self.fw = forward #预测后几天的收益
        self.slippage = 0.001 #滑点情况，共10个基点，买入卖出各一个基点
        
    def data_pre(self):
        '''
        数据准备，构建数据特征，并标记类别标签
        '''
        data = self.data
        data['retn'] = data['close'].shift(-self.fw) / data['open'] - 1
        attr = pd.DataFrame()
        #构建四个个数据特征，用于刻画K线的形态，并考虑交易量的变化
        attr['high_low'] = data['high'] / data['low'] - 1
        attr['high_close'] = data['high'] / data['close'] - 1
        attr['close_low'] = data['close'] / data['low'] - 1
        attr['close_open'] = data['close'] / data['open'] - 1
        attr['vol_pct'] = data['volume'].pct_change()
        attr.dropna(inplace= True)
        
        #将特征标准化
        for field in attr.columns:
            attr[field] = (attr[field] - attr[field].mean()) / attr[field].std()       
        
        attrs = {}
        dates = attr.index
        bk = self.bk
        #用前3天的特征值来预测后3天的涨跌涨跌方向
        for i in range(bk, len(attr) - bk + 1):
            attrs[dates[i-1]] = attr.iloc[i-bk : i, :].stack().values
        attrs = pd.DataFrame(attrs).T
        attrs['change_fw'] = data['close'].shift(-self.fw) - data['close']
        #上涨标记为1，下跌标记为-1
        attrs['label'] = np.where(attrs['change_fw'] > 0, 1, -1)
        attrs.drop('change_fw', axis=1, inplace=True)
        return attrs
    
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
        C_list = list(range(1, 10))
        gamma_list = np.linspace(0.01, 0.1, 10)
        (C, gamma), score = self.gscv_para(C_list, gamma_list, x_train, y_train) 
        #可能存在参数在边界的情况，此时需要重置一下参数的备选范围
#        if C == 1 or C == 10:
#            if C == 1:
#                print('C touched the lower limit')
#                C_list = np.linspace(0.1, 0.5, 5)
#            else:
#                print('C touched the upper limit')
#                C_list = list(range(10, 15))
#            C, gamma = self.gscv_para(C_list, gamma_list, x_train, y_train)    
#        else:
#            pass
#        if gamma == 0.01 or gamma == 0.1:
#            if gamma == 0.01:
#                print('gamma touched the lower limit')
#                gamma_list = np.linspace(0.001, 0.01, 10)
#            else:
#                print('gamma touched the upper limit')
#                gamma_list = np.linspace(0.1, 0.5, 9)
#            C, gamma = self.gscv_para(C_list, gamma_list, x_train, y_train)
#        else:
#            pass
        return C, gamma, score
    
    def predict(self, begin_year):
        '''
        用拟合的分类器对样本外的数据做预测
        '''
        train_data, test_data = self.data_split(begin_year)
        x_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        x_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
        C, gamma, score = self.svm_fit(x_train, y_train)
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
                    retns[date] = abs(data.loc[date, 'retn']) * (1-self.slippage)
                else:
                    #预测失误，产生损失
                    pct = data.loc[date, 'retn'] * (1+self.slippage)
                    retns[date] = pct if pct < 0 else -pct
        return paras, 1 + pd.Series(retns).sort_index().cumsum()
        
if __name__ == '__main__':

    import matplotlib.pyplot as plt    
    
    data = pd.read_excel('E:/Data/HS300_05_18.xlsx', index_col='date')
    SVM.data = data
    hs = SVM(3, 3)
    years = int(len(data) / 240) + 1
    paras, nav = hs.cum_retn(years - SVM.train_years)
    def drawdown(nav):
        #计算最大回撤
        dd = []
        for i in range(1, len(nav)):
            max_i = max(nav[:i])
            dd.append(min(0, nav[i] - max_i) / max_i)
        return dd
    Drawdown = pd.Series(drawdown(nav), index=nav.index[1:])
    maxdd = min(Drawdown) 
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(nav, label='strategy')
    hs300 = data['close'][-939:] / data['close'][-939]
    ax1.set_xlim(nav.index[0], nav.index[-1])
    ax1.plot(hs300, label='HS300')
    ax1.set_ylabel('Net Asset Value', fontdict={'fontsize':16})
    ax1.set_xlabel('Date', fontdict={'fontsize':16})
    ax1.legend(loc='center right', fontsize=16)
    ax2 = ax1.twinx()
    ax2.set_ylim(-1.5, 0)
    ax2.plot(Drawdown, color='c')
    ax2.set_ylabel('Max Drawdown', fontdict={'fontsize':16})
    ax2.fill_between(Drawdown.index, Drawdown, color='c')
    ax2.set_ylim(-1.5, 0)
    plt.savefig('svm_hs300.png', bbox_inches='tight')
    
    
    
    
