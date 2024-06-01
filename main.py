import numpy as np

# a = np.array([1,2,3,4])

# b = np.array([[0,1,2], [4,5,6]])
# print(b.shape)
# print(b)

# # Creatinng a 3x3 array of all zeros
# zeros = np.zeros((3,3))
# print("array of all zeros")
# print(zeros)

# # creating an array of all ones
# ones = np.ones((2,3))
# print('array of all ones')
# print(ones, ones.dtype)

# # creating a 3x3 with a constant values
# consValues = np.full((3,3), 7)
# print("array of constant values")
# print(consValues, consValues.dtype)

# # Creating an array with a random values
# print("array with random values")
# rndm = np.random.random((3,3))
# print(rndm)

# #creating 3x3 identity matrix
# identity = np.eye(3,3)
# print("identity matrix")
# print(identity)

# # creating an array using arange()
# myArray = np.arange(0,10)
# print("Range of array element")
# print(myArray)

# # array with linspace()
# '''the linspace() provide an array of specified range of numbers with the specified number of element'''
# print('array with intervals')
# lsp = np.linspace(1., 10., 7)
# print(lsp)


# x=np.array([[1,2,1], [2,1,2], [0,1,1]])
# y=np.array([[7,8,0], [0,1,2], [1,1,0]])

# print(x, "X")
# print(y, "Y")

# print(np.dot(x,y))

# the empty_like() helps to create an array same as the one passed as the parameter of the method

# print(np.reshape(x, (1,9)))
# print(x)


#******************************************* PANDAS *********************************************


import pandas as pd

# data = {
#     'Gender': ['M', 'F', 'Other'],
#     'ID_NO.': ['DAW-001', 'DAW-002', 'DAQ-003'],
#     'AGE': [22, 40, 35]
# }

# d = pd.DataFrame(data, columns=['ID_NO.', 'Gender', 'AGE'])
vgsale = pd.read_csv('vgsales.csv')
# print(vgsale)

# the describe() will return the quick statistical summary on dataFrame. But only for the column that contains numerical values
# print(vgsale.describe())
# the cov() shows how variables are related, the +ve cavarrience means thy're positively related,
# while -ve covarrience means they're negatively related
# print(vgsale.cov())
# """Furthermore: the corr() still tells how variables are positively or negatively related, 
#     but it also tells the degree where
#     the variables are tends to move"""
# print(vgsale.corr())

# print(vgsale.columns)
# print(vgsale[vgsale['Name'].duplicated()].count())

# print(vgsale['duplicate'])


# data_one = {
#  'emp_id': ['1', '2', '3', '4', '5'],
#  'first_name': ['Jason', 'Andy', 'Allen', 'Alice', 'Amy'],
#  'last_name': ['Larkin', 'Jacob', 'A', 'AA', 'Jackson']}

# df1 = pd.DataFrame(data_one)

# data_two = {
#  'emp_id': ['4', '5', '6', '7'],
#  'first_name': ['Brian', 'Shize', 'Kim', 'Jose'],
#  'last_name': ['Alexander', 'Suma', 'Mike', 'G']}

# df2 = pd.DataFrame(data_two)

# concatinating DataFrame
# df = pd.concat([df1, df2])
# print(df)

# print("*****************************")
# concatinating using append()
# df = df1.append(df2)
# print(df)

# merging two DataFrames base on the specific column

# merge = pd.merge(df1, df2, on=['emp_id'], how='left')
# print(merge)


#******************************************** MATPLOTLIB *********************************************************


import matplotlib .pyplot as plt

x = np.arange(5)
y = (12, 14, 20, 32, 10)

# plt.hist(x, y)
# plt.show()

# vgsale.plot()
# plt.show()

k = np.linspace(0, 20, 100)
v = np.sin(k)

# Customise axis label
# plt.plot(k, v, label='Sample label')
# plt.title('Sample Plot Label')
# plt.xlabel('x-axis label')
# plt.ylabel('y-axis lablel')
# plt.grid(True)
# plt.legend(loc='best')
# plt.tight_layout(pad=1)
# plt.savefig('my_figure.jpg')

# Line plot using  ax.plot()
m = np.linspace(0, 20, 100)
n = np.sin(m)

# fig = plt.figure(figsize=(8,4))
# ax = fig.add_subplot(1,1,1)
# ax.plot(m, n, 'b-', linewidth=2, label='sample label')
# ax.set_xlabel('X-axis label')
# ax.set_ylabel('Y-axis label')
# ax.legend(loc='best')
# ax.grid(True)
# fig.suptitle('Sample Plot Title')
# fig.tight_layout(pad=1)

# Second plot on same fig
z = np.arange(0,10,2)
w = np.arange(10, 0, -2)

# ax2 = fig.add_subplot(1,2,1)
# ax2.plot(l, w, linewidth=2, label='sample 2')
# ax2.set_xlabel('x-axis')
# ax2.set_ylabel('y-axis')
# ax2.legend(loc='best') 



# Multiple lines on same axis
#get fig and axe at once
fig, ax = plt.subplots(figsize=(8,4))

x1 = np.linspace(0, 100, 20)
x2 = np.linspace(0, 100, 20)
x3 = np.linspace(0, 100, 20)

y1 = np.sin(x1)
y2 = np.cos(x2)
y3 = np.tan(x3)

ax.plot(x1,y1, label='sin')
ax.plot(x2,y2, label='cos')
ax.plot(x3,y3, label='tan')

ax.grid(True)
ax.legend(loc='best')

#multiple line on different axis
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(8,4))
x1 = np.linspace(0, 100, 20)
x2 = np.linspace(0, 100, 20)
x3 = np.linspace(0, 100, 20)

y1 = np.sin(x1)
y2 = np.cos(x2)
y3 = np.tan(x3)

ax1.plot(x1, y1, label='sin', color='red')
ax2.plot(x2, y2, label='cos', color='blue')
ax3.plot(x3, y3, label='tan', color='green')

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

ax1.legend(loc='best', prop={'size':'large'})
ax2.legend(loc='best', prop={'size':'large'})
ax3.legend(loc='best', prop={'size':'large'})

fig.suptitle('A Simple Multi Axis Line Plot')

plt.show()
