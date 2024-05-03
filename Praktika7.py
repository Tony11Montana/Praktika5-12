import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from praktika6 import Praktika6


class Praktika7(Praktika6):
    def __init__(self, fileName, sep=','):
        self.fileName = fileName
        self.sep = sep
        self.data = pd.read_csv(fileName, sep=sep)

    def showDataTypes(self):
        print(self.data.dtypes)

    def showData(self):
        print(self.data)

    def multiplication(self, nameClmn, factor):
        self.data[nameClmn] = self.data[nameClmn].mul(factor)

    def maxValue(self, nameClmn):
        print(f'max value in columns - {nameClmn} = {self.data[nameClmn].max()}')
        print(self.data.loc[[self.data[nameClmn].idxmax()]])

    def minValue(self, nameClmn):
        print(f'min value in columns - {nameClmn} = {self.data[nameClmn].min()}')
        print(self.data.loc[[self.data[nameClmn].idxmin()]])

    def filNaInColumns(self, nameClmn, type='median'):
        obj = Praktika6(self.fileName, self.sep)
        obj.filNaInColumns(nameClmn, type)
        self.data[nameClmn] = obj.data[nameClmn]
    def countNaVal(self):
        print(self.data.isna().sum())
    def showGraph(self, columnOne, columnTwo):
        plt.scatter(self.data[columnOne], self.data[columnTwo])
        plt.xlabel(columnOne)
        plt.ylabel(columnTwo)
        plt.show()
    def correlForCSV_NBA(self):
        print(self.data[['Wingspan', 'Weight', 'Height']].corr())

    def LineRegr(self, columnOne, columnTwo):
        kf = np.polyfit(self.data[columnOne].values, self.data[columnTwo].values, deg=1)
        #b0 + ax
        x = self.data[columnOne].values
        y = kf[0] + kf[1] * x
        plt.plot(x,y)
        plt.title(f'Линейная регрессия y=${kf[0]}$+{kf[1]}*x')
        plt.xlabel(columnOne)
        plt.ylabel(columnTwo)
        plt.show()
    def newClmnFromCurrentClmn(self, nameNewClmn, columnOne, columnTwo):
        self.data[nameNewClmn] = self.data[columnTwo] - self.data[columnOne]
    def indexWeight(self, nameColumn):
        self.data.insert(len(self.data.columns), 'Index'+nameColumn, 0) # Add new Column
        for i, item in enumerate(self.data[nameColumn].sort_values().index):
            self.data.loc[item, 'Index'+nameColumn] = i
    def addictionColumns(self, nameColumn):   # Зависимости столбцов от позиции
        grouped = self.data.groupby('Position')
        career_mean = grouped['Career length in Years'].mean()
        height_mean = grouped['Height'].mean()
        weight_mean = grouped['Weight'].mean()
        arm_span_mean = grouped['Wingspan'].mean()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        career_mean.plot(kind='bar', ax=axes[0, 0], title='Average Career Length by Position')
        height_mean.plot(kind='bar', ax=axes[0, 1], title='Average Height by Position')
        weight_mean.plot(kind='bar', ax=axes[1, 0], title='Average Weight by Position')
        arm_span_mean.plot(kind='bar', ax=axes[1, 1], title='Average Arm Span by Position')
        plt.tight_layout()
        plt.show()
    def addictionFromTime(self):

        grouped = self.data.groupby('Year Start')
        mean_height = grouped['Height'].mean()
        mean_weight = grouped['Weight'].mean()
        mean_lenCareer = grouped['Career length in Years'].mean()
        mean_wingspan = grouped['Wingspan'].mean()

        #Graphic
        figure, ax = plt.subplots(2,2, figsize=(12, 12))
        mean_height.plot(kind = 'bar', ax=ax[0,0], title= 'Mean height by time', xlabel= 'time', ylabel='mean height')
        mean_weight.plot(kind = 'bar', ax=ax[0,1], title= 'Mean weight by time', xlabel= 'time', ylabel='mean weight')
        mean_lenCareer.plot(kind='bar', ax=ax[1,0], title= 'Mean lenCareer by time', xlabel= 'time', ylabel='mean lenCareer')
        mean_wingspan.plot(kind='bar', ax=ax[1, 1], title='Mean wingspan by time', xlabel='time',
                            ylabel='mean wingspan')
        plt.show()

if __name__ == '__main__':
    df = Praktika7(fileName='NBA.csv')
    df.showData()
    df.showDataTypes()
    df.filNaInColumns('Wingspan', 'median')
    df.filNaInColumns('Weight', 'median')
    df.multiplication('Weight', 0.45359237)
    df.maxValue('Height')
    df.minValue('Height')
    df.maxValue('Weight')
    df.minValue('Weight')
    df.maxValue('Wingspan')
    df.minValue('Wingspan')
    df.countNaVal()
    #5 NUMBER.
    df.correlForCSV_NBA()
    df.showGraph('Height', 'Wingspan')
    df.showGraph('Weight', 'Wingspan')
    df.showGraph('Height', 'Weight')
    df.LineRegr('Weight', 'Height')
    df.newClmnFromCurrentClmn('Career length in Years','Year Start','Year End')
    df.showData()
    df.indexWeight('Weight')
    df.showData()
    df.addictionColumns('nameColumn')
    df.addictionFromTime()