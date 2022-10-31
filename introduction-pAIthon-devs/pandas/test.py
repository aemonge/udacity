import pandas as pd;
import numpy as np;

def bananas():
    groceries = pd.Series(data=[30, 6, 'yes', 'no'], index=['eggs', 'apples', 'milk', 'bread'])
    if ('banana' not in groceries.index):
        print(groceries.index)
    print(groceries)

    print(groceries[['milk', 'eggs']], '\n')
    print(groceries[1:])
    print(groceries['apples'], '\n')

    groceries['apples'] = 30 # ERROR: groceries['apples'] + 20 # Error
    groceries.drop('milk')
    print(groceries, '\n')
    groceries.drop('milk', inplace=True)
    print(groceries)

def fruits():
    fruits = pd.Series([10, 6, 3], ['Apples', 'Bananas', 'Oranges'])
    fruits['Bananas'] = fruits['Bananas'] + 2 # type: ignore
    print(fruits)
    fruits = fruits + 5
    print(fruits)
    fruits = np.power(fruits, 2)
    print(fruits)

def spreadsheet():
    data = dict({
        'Bob': pd.Series(index=['Bike', 'Car', 'Scooter'], data=[int(300), int(15000), int(200)]), # Why 🖐️ U do this
        'Alice': pd.Series(index=['Car', 'House', 'Pot'], data=[int(12000), int(5000000), int(20)]) # I want INT
    })
    items = pd.DataFrame(data)
    print(items, '\n')
    print(items.columns, items.index, '\n')
    print(pd.DataFrame(data, index=['Car', 'Bike']))
    print(pd.DataFrame(data, columns=['Bob']), '\n')

    print(items[['Bob']]) # items.columns['Bob']
    print(items.loc[['Bike']], '\n') # items.columns['Bob']

    items['Rob']  = [150, 1200, -1, 20, -1] # I can't NaN nor Nil this 🙁
    items['Ob\'s Family'] = items['Bob'] + items['Rob'] # type: ignore
    print(items, '\n')
    items = items.append(pd.DataFrame([{'Bob': 3000, 'Alice': 5000}], index=["Kitchen"])) # 😮 .concat wont' work
    print(items, '\n')
    items.insert(2, 'Sofia', [-1, -1, 1000000, -1, -1, -1]) # 😮 I can't Nil, nor leave emtpy for Sofia
    # items.insert(3, 'Sofia', [{'Bike': 0, 'Car': 0, 'House': 1000000, 'Pot': 0, 'Scooter': 0, 'Kitchen': 0}])
    print(items, '\n')

    items = items.rename(index={'Pot': 'Skillet'}, columns={'Ob\'s Family': 'Obson'})
    print(items, '\n')

    print(items.dropna(axis=0), '\n', items.dropna(axis=1), '\n')
    print(items.isnull())
    print('_______________________________________________________')
    items = items.fillna(-1)
    print(items, '\n')

def databases():
    data = pd.read_csv('./fake-company.csv')
    print(data)
    print(data.groupby(['Year'])['Salary'].sum())
    print(data.groupby(['Department'])['Salary'].mean())

databases()
