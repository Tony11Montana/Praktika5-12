import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('world_population.csv')
    # print(data.head())
    # print(data.info())
    # print(data.shape) # (row , columns)
    # print(data.describe())
    # print(data.values)
    # print(np.count_nonzero(np.isnan(data.select_dtypes(include='number')))) # Check on value Nan with help library numpy
    # # Разведовательный анализ.
    # sns.pairplot(data=data.select_dtypes('number'))
    # sns.set(style='ticks')
    # plt.show()
    # sns.heatmap(data=data.select_dtypes('number'))
    # plt.show()
    # sns.scatterplot(data=data, y='World Population Percentage', x='2022 Population')
    # plt.show()
    # fig, ax = plt.subplots(2,2, figsize=(14,10))
    # hist_data = data.select_dtypes(include='number')
    # hist_data.drop('Rank', inplace=True, axis=1)
    # sns.histplot(data=hist_data, x='Area (km²)', kde=True, ax=ax[0,0])
    # sns.histplot(data=hist_data, x='Growth Rate', kde=True, ax=ax[0, 1])
    # sns.histplot(data=hist_data, x='Density (per km²)', kde=True, ax=ax[1, 0])
    # sns.histplot(data=hist_data, x='2022 Population', kde=True, ax=ax[1, 1])
    # plt.show()
    # sns.boxplot(data=data, x='Growth Rate')
    # plt.show()
    # sns.violinplot(data=data, x='Continent', y='2022 Population')
    # plt.show()
    #4.
    # corr_matrix = data.select_dtypes(include='number').corr()
    # df_corr = pd.DataFrame(corr_matrix[corr_matrix > 0.7])
    # print(df_corr)
    # sns.heatmap(df_corr, annot=True, cmap='coolwarm')
    # plt.title('Корреляция больше 0.7')
    # plt.show()

    #Dynamic.
    grouped = data.groupby('Continent')[['2022 Population', '2020 Population', '2015 Population', '2010 Population', '2000 Population', '1990 Population', '1980 Population', '1970 Population']].sum()
    grouped.plot(kind='bar', title='Динамика населения на каждом континенте.')
    plt.show()