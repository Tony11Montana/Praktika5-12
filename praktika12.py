import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# sklearn.datasets.load_iris
# data = pd.read_csv('Iris.csv')
# data.drop('Id', inplace=True, axis=1)
# print(data.head(5))
# x = data.iloc[:,:-1]
# y = data['Species']
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.20, random_state=27)
# print('X_train')
# print(X_train)
# print('y_train')
# print(y_train)
# SVC_model = sklearn.svm.SVC()
# KNN_model = KNeighborsClassifier(n_neighbors=5)
# SVC_model.fit(X_train, y_train)
# KNN_model.fit(X_train, y_train)
# SVC_prediction = SVC_model.predict(X_test)
# KNN_prediction = KNN_model.predict(X_test)
# print('Оценка точности')
# print('Оценка точности SVC', accuracy_score(SVC_prediction, y_test))
# print('Оценка точности KNN', accuracy_score(KNN_prediction, y_test))
# print('матрица неточности')
# print(confusion_matrix(SVC_prediction, y_test))
# print('отчет о классификации')
# print(classification_report(KNN_prediction, y_test))


def DropColmWithLowCorr(data):
    dfCorr = data.corr()
    print(dfCorr)
    # Для нахождения столбцов не имеющих значения для обучения, применяется среднее значение столбца из корреляционной матрицы.
    dataWithLowCorr = abs(dfCorr.mean())
    # Сортировка по возрастанию.
    dataWithLowCorr.sort_values(inplace=True)
    print(dataWithLowCorr)
    # Нахождение квантиля 30%.
    print(dataWithLowCorr.quantile(0.30))
    dataWithLowCorr = dataWithLowCorr[dataWithLowCorr <= dataWithLowCorr.quantile(0.30)]

    data = data.drop(dataWithLowCorr.index, axis=1)
    return data

def Analysis():
    # #1 Парные диаграммы.
    # sns.pairplot(data)
    # plt.title('Парные диаграммы')
    # plt.show()
    #2 Тепловая карта корреляции.
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта корреляции')
    plt.show()
    # #3 Гистограммы.
    # sns.pairplot(kind='hist', data=data)
    # plt.title('Гистограммы')
    # plt.show()
    # #4 Ящик с усами.
    # sns.boxplot(data)
    # plt.title('Ящик с усами')
    # plt.show()

if __name__ == '__main__':
    data = pd.read_csv('heart.csv')

    target = data['output']

    #Data info.
    print(f'row - {data.shape[0]}, column - {data.shape[1]}')

    print(data.info())

    print(data.head())

    print(data.describe())

    #isNaN
    print('Values is NaN :')
    SeriesNoZero = data.isna().sum()
    print(SeriesNoZero[SeriesNoZero > 0])

    Analysis()

    data = DropColmWithLowCorr(data)
    print(data)

    # EDUCATION.
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()

    X_scaler = scaler.fit_transform(data)

    print(data)

    print(type(X_scaler))

    x = pd.DataFrame(X_scaler, columns=data.columns)

    print(target.value_counts())

    # Разделение набора на обучающую и тестовую выборку.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, target,
                                                                                test_size=0.20, random_state=27)

    # 1. Метод опорных векторов (Support Vector Machine)
    SVC_model = sklearn.svm.SVC()
    SVC_model.fit(X_train, y_train)
    SVC_predict = SVC_model.predict(X_test)
    print(f'accuracy_score SVC = {accuracy_score(SVC_predict, y_test)}')

    from sklearn.linear_model import LogisticRegression
    # 2. Логистическая регрессия (LogisticRegression)
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    print(f'accuracy_score classification = {accuracy_score(logistic_pred, y_test)}')

    from sklearn import tree
    # 3. Дерево решений (Decision Tree)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    clf_predict = clf.predict(X_test)
    print(f'accuracy_score CLF = {accuracy_score(clf_predict, y_test)}')

    from sklearn.ensemble import RandomForestClassifier
    # 4. Случайный лес (RandomForest)
    forest = RandomForestClassifier()
    rfc = forest.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    print("Random Forest accuracy:", accuracy_score(rfc_predict, y_test))

    #Вывод , модель классификации подходит лучше всего для прогнозирования инфаркта.