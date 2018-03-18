# -*- coding: utf-8 -*-
#利用本框架进行ai模型测试
from __future__ import print_function, absolute_import
from gm.api import *

import trade_model_v3 as tm
import time
import gmTools_v2 as gmTools

eps = 1

class Mystrategy():
    def __init__(self, *args, **kwargs):
        self.buy_list=[]  #字典列表
        self.sell_list=[]
        self.trade_count=0
        self.trade_limit=5
        self.now=''
        self.positions =''
        self.holding_list=[]
        self.sells=''

    def on_bar(self, context):
        return

        print("[%s]on_bar " % (time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time()))))

        self.positions = self.get_positions()  # 查询策略所持有的多仓
        # 打印持仓信息
        # self.print_position()
        self.sells = [x['code'] for x in self.sell_list]

        stop_time = context.now
        self.now=stop_time
        #一次读取多日或多周期的K线数据进行分析，提交整体运行效率，
        # 买卖处理也采用自行定义方式，掘金量化仅提供所需的基础数据
        self.buy_list,self.sell_list=tm.get_bs_list(stop_time)

        # 打印持仓信息
        self.print_position()

        self.trade(bar.exchange+ '.'+ bar.sec_id,bar)

    #TODO  持仓股价沟通完毕才打印持仓信息
    def print_position(self):
        positions = self.get_positions()  # 查询策略所持有的多仓
        if  len(positions) > 0:
            now = '[{0}]'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print(now,self.now,'holding ...')
            holding=0
            for position in positions:
                stock=position.exchange+'.'+ position.sec_id
                self.holding_list.append(stock)
                price=self.get_last_n_bars(stock,60,1,self.now)[0].close
                reward = price * 100 / position.vwap-100
                tmp=position.volume*price
                print("     code:%s,vol:%d,amount:%.2f,reward:%.2f" % (
                    stock, position.volume,tmp , reward))
                holding+=tmp
            cash=self.get_cash()
            print("     nav:%.2f,av:%.2f,hold:%.2f,reward:%.2f%%\n"
                  % (cash.nav,cash.available,holding,cash.profit_ratio ))

    #交易处理有关函数
    def trade(self,sec,bar):
        if sec in self.holding_list:
             for position in self.positions:
                stock = position.exchange + '.' + position.sec_id
                if  stock==sec:
                    if position.available_yesterday > eps :
                        if stock in self.sells :
                            # sell out stock in sell list
                            self.close_long(position.exchange, position.sec_id,
                                            0, position.available_yesterday)
                        else:
                            price = self.get_last_n_bars(stock, 60, 1, self.now)[0].close
                            reward = price * 100 / position.vwap - 100
                            if reward<-8:  #stop loss
                                # sell out
                                self.close_long(position.exchange, position.sec_id,
                                                0, position.available_yesterday)

                    break

        if len(self.buy_list)>0:
            for i in  range(len(self.buy_list)):
                item=self.buy_list[i]
                stock = item['code']

                # 没有超出下单次数限制
                if stock == sec:
                    if len(self.positions) < self.trade_limit :
                        #a stock only can buy once
                        holding=False
                        for position in self.positions:
                            if sec==position.exchange+'.'+position.sec_id:
                                holding=True
                                break

                        if holding==False:
                            price = bar.close
                            cash=self.get_cash()
                            vol = int(cash.available/
                                    ((self.trade_limit-len(self.positions) )*price*130))
                            # 如果本次下单量大于0,  发出买入委托交易指令
                            if vol >= eps:
                                order=self.open_long(stock[:4], stock[5:], price, vol*100)

                    break

mystrategy=Mystrategy()

def init(context):
    # 每天14:50 定时执行algo任务
    schedule(schedule_func=algo, date_rule='daily', time_rule='14:50:00')


def algo(context):
    global  mystrategy

    mystrategy.on_bar(context)


# 查看最终的回测结果
def on_backtest_finished(context, indicator):
    print(indicator)



if __name__ == '__main__':
    securities = tm.get_stock_list()
    tm.backtest_model()
    run(strategy_id='a3adbebf-04df-11e8-b3f7-dc5360304926',
        filename='aiTrade.py',
        mode=MODE_BACKTEST,
        token='c631be98d34115bd763033a89b4b632cef5e3bb1',
        backtest_start_time='2018-01-01 09:30:00',
        backtest_end_time='2018-08-21 15:00:00')