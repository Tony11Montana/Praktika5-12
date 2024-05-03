import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def ex4():
    mean_val = data['Rating'].mean()
    std_val = data['Rating'].std()

    print(f'STD = {std_val}')
    print(f'MEAN = {mean_val}')
    print(f'Up border = {mean_val + std_val}')
    print(f'Down border = {mean_val - std_val}')

    grouped = data.groupby('Title')
    mean_rating = grouped['Rating'].mean()
    mean_rating.name = 'Average Rating'
    print('00000000000000000000000000000000000000000000000')
    #mean_rating_df = mean_rating.rename('Mean Rating').reset_index()
    print(mean_rating[(mean_rating >= mean_val + std_val) | (mean_rating <= mean_val - std_val)])
def ex3():
    table = data.pivot_table(values='Rating',columns='Gender', index='Title', aggfunc='mean', fill_value=0.)
    table['Diff'] = abs(table['F'] - table['M'])
    print(table.sort_values('Diff'))
def ex2():
    table = data.pivot_table(values='Rating',columns='Gender', index='Title', aggfunc='mean', fill_value=0.)
    print(table)
def ex1():
    dataEx1 = data[data['Title'] == 'Communion (1989)']
    groupedGender = dataEx1.groupby('Gender')
    groupedAge = dataEx1.groupby('Age')
    mean_rating = groupedAge['Rating'].mean()
    print(mean_rating)
    mean_rating = groupedGender['Rating'].mean()
    print(mean_rating)

if __name__ == '__main__':
    mov = pd.read_csv('movies.csv', sep='::', names=['MovieID', 'Title', 'Genres'])
    mov['MovieID'].astype(int)
    users = pd.read_csv('users.csv', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    ratings = pd.read_csv('ratings.csv', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    merge = pd.merge(users, ratings, on='UserID', how='inner')
    data = pd.merge(mov, merge, on='MovieID', how='inner')
    #print(mov.shape[0] + merge.shape[0])
    #print(data)
    #sns.relplot(data=mov, x='Genres', y='Title')
    #sns.relplot(data=mov)
    #plt.show()
    print('Exercises---------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')
    ex1()
    ex2()
    ex3()
    ex4()
    print('Exercises----------------------------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------------------------------')
