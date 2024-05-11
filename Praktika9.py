import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def interval(dataStart, dataEnd):
    dataWithInterval = data[data['Date'].between(dataStart, dataEnd)]
    return dataWithInterval
def lineGraph(x,y, tle):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(tle)
    plt.xlabel('Date')
    plt.ylabel(y.name)
    plt.show()

def ex8(dataStart, dataEnd):
    data8 = interval(dataStart, dataEnd)
    grouped = data8.groupby(data8.Date.dt.month)
    moving_ave_adjClose = grouped['Adj Close'].mean()
    exp_ave_adjClose = grouped['Adj Close'].ewm(span=20, adjust=False).mean()
    moving_ave_adjClose.plot(label='moving_ave_adjClose')
    exp_ave_adjClose.plot(label='exp_ave_adjClose')
    plt.title('Moving Average and Exponential Moving Average for Adj Close')
    plt.xlabel('Month')
    plt.ylabel('Adj Close')
    plt.legend()
    plt.show()
def diagramGraph(y, tle):
    fig = plt.figure()
    #plt.bar(x, y)
    #plt.title(tle)
    #plt.xlabel('Date')
    #plt.ylabel(y.name)
    plt.hist(y, bins=y.shape[0])
    plt.title(tle)
    plt.xlabel(y.name)
    plt.ylabel('Частота')
    plt.show()
def ex3(dataStart, dataEnd,y, nameGraph):
    data3 = interval(dataStart, dataEnd)
    diagramGraph(data3[y], nameGraph)
def ex2(dataStart, dataEnd, x, y):
    data2 = interval(dataStart, dataEnd)
    lineGraph(data2[x], data2[y], tle='Price for date')
def ex7(dataStart, dataEnd, x, y):
    data2 = interval(dataStart, dataEnd)
    lineGraph(data2[x], data2[y], tle="Volumes")
def ex1(dataStart, dataEnd, x, y):
    data1 = interval(dataStart, dataEnd)
    lineGraph(data1[x], data1[y], tle='History price')
def ex4(dataStart, dataEnd, y, y1, cum = False, g = False):
    if g:
        orient = 'horizontal'
    else:
        orient = 'vertical'
    data4 = interval(dataStart, dataEnd)
    fig, ax = plt.subplots(2,1, figsize=(12,20))
    ax[0].hist(data4[y], bins='auto', orientation= orient)
    ax[0].set_title(f'Гистограмма цены {y} во временном промежутке {dataStart} - {dataEnd}')
    ax[1].hist(data4[y1], bins='auto', cumulative=cum)
    ax[1].set_title(f'Гистограмма цены {y1} во временном промежутке {dataStart} - {dataEnd}')
    plt.show()
def ex9(dataStart, dataEnd):
    data9 = interval(dataStart, dataEnd)
    data9['ROI'] = (data9['Close'] / (data9['Open']/100) ) - 100
    plt.bar(data9['Date'], data9['ROI'])
    plt.title('Дневная доходность акции в %', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('ROI, %')
    plt.show()
if __name__ == '__main__':
    data = pd.read_csv('alphabet_stock_data.csv')
    print(data)
    data['Date'] = pd.to_datetime(data['Date'])
    print(data.dtypes)
    print(data.sort_values('Date'))
    ex1('2020-05-10', '2020-09-20', x='Date', y='Adj Close') #History price.
    ex2('2020-05-10', '2020-09-20', x='Date', y='Open') # Open.
    ex2('2020-08-10', '2020-09-20', x='Date', y='Close') # close.
    ex3('2020-08-10', '2020-09-20', y='Volume', nameGraph='Gistogramm volume')
    ex4('2020-08-10', '2020-09-20', y='Open', y1='Close')
    #5 number.
    ex4('2020-08-10', '2020-09-20', y='Open', y1='Close')
    ex4('2020-08-10', '2020-09-20', y='High', y1='Low')
    #6 number.
    ex4('2020-08-10', '2020-09-20', y='Open', y1='Open', cum=True, g=True)
    #7.
    ex2('2020-05-10', '2020-09-20', x='Date', y='Open')
    ex7('2020-05-10', '2020-09-20', x='Date', y='Volume')
    #8.
    ex8('2020-05-10', '2020-09-20')
    #9.
    ex9('2020-05-10', '2020-09-20')