import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def VisualStatistiks():
    grouped_platform = data.groupby('Platform')['Sum_Sales'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(2, 2, figsize=(20, 18))
    for i, elem in enumerate(grouped_platform.index[0:4]):
        row = i//2
        clm = i%2
        dataForPlatf = data[data['Platform'] == elem]
        grouped_year_SumSales = dataForPlatf.groupby('Year_of_Release')['Sum_Sales'].sum()
        sns.barplot(grouped_year_SumSales, ax=ax[row, clm])
        ax[row, clm].set_title(f'Platform {elem}')
    #
    plt.suptitle('Гистограмма распределения выпущенных игр по годам на самых популярных платформах', fontsize=16)
    plt.show()

    grouped_year_agrCount = data.groupby('Year_of_Release')['Sum_Sales'].sum()
    grouped_genre = data.groupby('Genre')['Sum_Sales'].sum()
    sns.barplot(grouped_year_agrCount)
    plt.title('Гистограмма распределения выпущенных игр по годам', fontsize=16)
    plt.show()
    plt.pie(grouped_genre.values, labels=grouped_genre.index, autopct='%1.1f%%')
    plt.title('Pаспределение продаж по жанрам')
    plt.tight_layout()
    plt.show()

    # Создаем фигуру и подобласти рисования
    fig, ax = plt.subplots(2, 2, figsize=(20, 18))

    for i, elem in enumerate(grouped_platform.index[0:4]):
        row = i // 2
        clm = i % 2
        dataForPlatf = data[data['Platform'] == elem]
        sns.boxplot(data=dataForPlatf, x='Genre', y='Sum_Sales', ax=ax[row, clm])
        ax[row, clm].set_title(f'Platform {elem}')

    grouped_genre = grouped_genre.sort_values(ascending=False).head(10)
    plt.suptitle('Boxplots для продаж по жанрам на самых популярных платформах', fontsize=16)
    plt.tight_layout()
    plt.show()
    print('----------------------Very popular platforms----------------------------------------')
    print(grouped_platform.index)
    print('----------------------Very popular genres----------------------------------------')
    print(grouped_genre.index)


if __name__ == '__main__':
    #1-2.
    data = pd.read_csv('games.csv')
    print(data.head())
    print(data.info())
    print(data.isna().sum())
    data.dropna(subset=['Name', 'Year_of_Release'], inplace =True)
    print(data.isna().sum())
    data['Year_of_Release'] = data['Year_of_Release'].astype(copy=False, dtype='int64')
    print('Was NaN value----------',data['User_Score'].isna().sum())
    data['User_Score'] = data['User_Score'].replace('tbd', np.NaN)
    data['User_Score'] = data['User_Score'].astype('float64')
    print('Now NaN value----------', data['User_Score'].isna().sum())
    data['User_Score'] = data['User_Score'].replace(np.NaN, 'Unknown')
    #Add new Column with Sum Sales
    data['Sum_Sales'] = data['NA_sales'] + data['EU_sales'] + data['JP_sales'] + data['Other_sales']
    VisualStatistiks()