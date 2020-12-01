# refer to
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# from pandas import read_csv
from pandas import read_excel
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Load dataset
# filePath = "C:\\phillipWorks\\hud\\myFramework\\Dataset\\Iris\\iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(filePath, names=names)
filePath = "C:\\phillipWorks\\hud\\myFramework\\Dataset\\Iris\\iris.xlsx"
names = ['Id', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_excel(filePath, names=names)

print('========== Summarize the Dataset ==========')
# shape
print('---------- Dimensions of Dataset ----------')
print(dataset.shape)
print()
# head
print('---------- Peek at the Data ----------')
print(dataset.head(20))
print()
# descriptions
print('---------- Statistical Summary ----------')
print(dataset.describe())
print()
# class distribution
print('---------- Class Distribution ----------')
print(dataset.groupby('class').size())
print()

print('========== Data Visualization ==========')
# box and whisker plots
print('---------- Univariate Plots ----------')
dataset.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
pyplot.show()
print()
# histograms
print('---------- Histogram Plots ----------')
dataset.hist()
pyplot.show()
print()
# scatter plot matrix
print('---------- Multivariate Plots ----------')
scatter_matrix(dataset)
pyplot.show()
print()
