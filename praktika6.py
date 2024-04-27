import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Praktika6:
    def __init__(self, pathOfFile, sep=','):
        self.data = pd.read_csv(pathOfFile, sep=sep)

    def showData(self, nameClmn=''):
        if nameClmn == '':
            print(self.data)
        else:
            print(self.data[nameClmn])

    def filNaInColumns(self, nameClmn, type='median'):
        if type == 'median':
            self.data[nameClmn] = self.data[nameClmn].fillna(self.data[nameClmn].median())
        elif type == 'max':
            max = (self.data[nameClmn].value_counts()).idxmax()
            self.data[nameClmn] = self.data[nameClmn].fillna(max)

    def countNaVal(self):
        print('Пустые значения в столбцах:')
        print(self.data.isna().sum())

    def showDataTypes(self):
        print(self.data.dtypes)

    def replaceValueOnInt(self, nameClmn):
        countVal = self.data[nameClmn].unique()
        countNumber = np.arange(0, len(countVal), 1)
        self.data[nameClmn] = self.data[nameClmn].replace(countVal, countNumber)

    def statisticksGender(self, gender, addictionColumn=''):
        if addictionColumn == '':
            if gender == 'male':
                print(self.data[self.data['Sex'] == 0].describe())
            elif gender == 'female':
                print(self.data[self.data['Sex'] == 1].describe())
        else:
            if gender == 'male':
                plt.bar((self.data['Pclass'][self.data['Sex'] == 0]).value_counts().index,
                        (self.data['Pclass'][self.data['Sex'] == 0]).value_counts().values)
                plt.title(f'Распределение {gender} по классам кают')
                plt.xlabel('Класс кают')
                plt.show()
                print(f'Распределение {gender} по классам кают')
                print((self.data['Pclass'][self.data['Sex'] == 0]).value_counts())
            elif gender == 'female':
                plt.bar((self.data['Pclass'][self.data['Sex'] == 1]).value_counts().index,
                        (self.data['Pclass'][self.data['Sex'] == 1]).value_counts().values)
                plt.title(f'Распределение {gender} по классам кают:')
                plt.xlabel('Класс кают')
                plt.show()
                print(f'Распределение {gender} по классам кают:')
                print((self.data['Pclass'][self.data['Sex'] == 1]).value_counts())

    def statisticksSurvivalRate(self):
        figure = plt.figure()
        plt.subplot(2, 2, 1)
        plt.bar((self.data['Pclass'].value_counts()).index, self.data['Pclass'].value_counts().values)
        plt.title('Статистика выживших от класса.')
        plt.xlabel('Класс')
        plt.subplot(2, 2, 2)
        plt.bar((self.data['Sex'].value_counts()).index, self.data['Sex'].value_counts().values)
        plt.title('Статистика выживших от пола.')
        plt.xlabel('Пол')
        plt.subplot(2, 2, 3)
        plt.bar((self.data['Age'].value_counts()).index, self.data['Age'].value_counts().values)
        plt.title('Статистика выживших от возраста.')
        plt.xlabel('Возраст')
        plt.subplot(2, 2, 4)
        self.data['Family'] = self.data['SibSp'] + self.data['Parch']
        plt.bar((self.data['Family'].value_counts()).index, self.data['Family'].value_counts().values)
        plt.title('Статистика выживших от наличия семьи на борту.')
        plt.xlabel('Наличие семьи на борту')
        plt.show()

    def sumValue(self, nameClmn):
        print(f'Всего {nameClmn} на борту:')
        print(self.data[nameClmn].sum())


if __name__ == '__main__':
    df = Praktika6('Titanik.csv')
    df.countNaVal()
    df.filNaInColumns('Age')
    df.filNaInColumns('Fare')
    df.showData()
    df.countNaVal()
    df.showDataTypes()
    df.replaceValueOnInt('Sex')
    df.showDataTypes()
    df.showData('Sex')
    df.replaceValueOnInt('Embarked')
    df.showData('Embarked')
    df.statisticksGender('female')
    df.statisticksSurvivalRate()
    df.sumValue('SibSp')
    df.sumValue('Parch')
    df.statisticksGender('male', 'Pclass')
    df.statisticksGender('female', 'Pclass')
