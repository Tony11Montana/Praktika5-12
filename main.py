import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ex1():
    data['Star color'] = data['Star color'].str.lower()
    data['Star color'] = data['Star color'].replace(to_replace='[^a-z\s]', value=' ', regex=True)
    print(data['Star color'])
def ex2():
    print(data)
    forAdd = []
    for i in range(data.shape[0]):
        n = data.at[i, 'Star type']
        if n == 0:
            forAdd.append('Red Dwarf')
            #data.at[i,'Star type'] = 'Red Dwarf'
        elif n == 1:
            forAdd.append('Brown Dwarf')
        elif n == 2:
            forAdd.append('White Dwarf')
        elif n == 3:
            forAdd.append('Main Sequence')
        elif n == 4:
            forAdd.append('Super Giants')
        elif n == 5:
            forAdd.append('Hyper Giants')
    data['Type of Star'] = forAdd
def ex3():
    forAdd = []
    for i in range(data.shape[0]):
        n = data.at[i, 'Spectral Class']
        if n == 'O':
            forAdd.append(0)
        elif n == 'B':
            forAdd.append(1)
        elif n == 'A':
            forAdd.append(2)
        elif n == 'F':
            forAdd.append(3)
        elif n == 'G':
            forAdd.append(4)
        elif n == 'K':
            forAdd.append(5)
        elif n == 'M':
            forAdd.append(6)
    data['Spectral Class int64'] = forAdd
def ex4(nameOfColumns):
    print(data[nameOfColumns].value_counts())
def ex5(nameOfColumn,columnForResult):
    array = data[nameOfColumn].unique()
    for item in array:
        select_data = data[data[nameOfColumn] == item]
        maxval = select_data[columnForResult].max()
        minval = select_data[columnForResult].min()
        meanval = select_data[columnForResult].mean()
        print('Statistics for each from:',nameOfColumn,'. Value: ',item)
        print('Max value ', maxval)
        print('Min value ', minval)
        print('Mean value ', meanval)
def ex6():
    data6 = data.select_dtypes(include= ['int64','float64'])
    print(data6.corr())
def ex1_1():
    data2['Date'] = data2['Year'].astype(str) + '-' + data2['Month'].astype(str) + '-' + data2['Day'].astype(str)
    data2['Date'] = pd.to_datetime(data2['Date'])
    #data2.drop(['Year','Month','Day'], inplace=True, axis=1)
    print(data2)
def ex2_2():
    data2.replace(-1,np.nan,inplace=True)
    print(data2)
def ex3_3():
    print(data2['Year'].value_counts())
    sum = 0
    arrSum = []
    arrCount = []
    for item in data2['Year'].unique():
        select_data2 = data2[data2['Year'] == item]
        select_data21 = data2[(data2['Year'] == item) & (data2['Standard Deviation'].notna())]
        count = select_data21.count()
        arrCount.append(count)
        sum = select_data2['Number of Sunspots'].sum()
        arrSum.append(sum)
        print('За',item,"год, солнечных пятен -",sum)
    graphForEx3_3(arrSum,arrCount)
def graphForEx3_3(arrSum,arrCount):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data2['Year'].unique(),arrCount)
    plt.ylabel("Кол-во дней")
    plt.title('Количество дней, за которое присутствуют данные в году ')
    plt.subplot(2,1,2)
    plt.bar(data2['Year'].unique(), arrSum)
    plt.xlabel("Год")
    plt.ylabel("Штук")
    plt.title('Сумма солнечных пятен')
    plt.show()
def ex4_4():
    arrDate = []
    arrValMean = []
    data2_4 = data2[data2['Year']//100 + 1 == 21]
    for item in data2_4['Year'].unique():
        for item1 in data2_4['Month'].unique():
            data4 = data2_4[(data2_4['Month'] == item1) & (data2_4['Year'] == item) & (data2_4['Number of Sunspots'].notna())]
            meanval = data4['Number of Sunspots'].mean()
            arrDate.append(str(item) + '/' + str(item1))
            arrValMean.append(meanval)
            print('среднее значение = ',meanval,'Mесяц =',item1,'Year =',item)
    plt.bar(arrDate,arrValMean)
    plt.title('Среднее кол-во пятен каждый месяц за 21 век.')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('stars.csv')
    ex1()
    ex2()
    ex3()
    #ex4(input("Enter name of column: "))
    #print(data['Star type'].unique())
    #print(data['Spectral Class int64'])
    #print(data['Star type'].max())
    ex5('Star type','Absolute magnitude(Mv)')
    ex5('Spectral Class', 'Temperature (K)') #6 ex
    ex6()
    #------------------------------------------------------------------------------------------
    data2 = pd.read_csv('sunspot.csv', delimiter=';')
    ex1_1()
    ex2_2()
    ex3_3()
    ex4_4()
