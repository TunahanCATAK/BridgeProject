import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import matplotlib.pyplot as plt


def Sigmoid(X, theta):
    h_theta = X @ theta.T
    denumerator = 1.0 + np.exp(-h_theta)
    return 1.0 / denumerator

def ComputeCost(X, y, theta):

    h_theta = Sigmoid(X, theta)

    log_sigmoid = np.log(h_theta)
    log_sigmoid_minus_one = np.log(1 - h_theta)

    mp_log_y = np.multiply(y, log_sigmoid)
    mp_log_y_minus_one = np.multiply(1-y, log_sigmoid_minus_one)

    return (-1*mp_log_y - mp_log_y_minus_one).mean()

def GradientDescent(X, y, theta, alpha, iters, X_test, y_test):
    cost = np.zeros(iters)
    test_cost = np.zeros(iters)
    for i in range(iters):
        total_cost = np.dot(X.T,(Sigmoid(X, theta) - y))
        theta = theta - (((alpha / y.shape[0]) * total_cost).T + 0.01* np.sum(theta))
        cost[i] = ComputeCost(X, y, theta)
        test_cost[i] = ComputeCost(X_test, y_test, theta)
        if i % 10 == 0: # just look at cost every ten loops for debugging
             print(cost[i], test_cost[i])
    return (theta, cost, test_cost)

def PlotCostValues(cost, test_cost):
    import matplotlib.pyplot as plt

    df = pd.DataFrame({'cost': cost, 'test_cost': test_cost})
    plt.plot('cost', data=df, color='red')
    plt.plot('test_cost', data=df, color='blue')
    plt.legend()
    plt.show()



"""
1st step : Read Data from a csv file.
"""
data = pd.read_csv('SampleData.csv', encoding='iso-8859-9')

"""
2nd step : Convert DateTime to a numeric format.
"""
hours = pd.to_datetime(data['Zaman'], format='%H:%M').dt.hour
minutes = pd.to_datetime(data['Zaman'], format='%H:%M').dt.minute

#data['Zaman'] = hours * 60 + minutes
data['Zaman'] = (hours - 8) * 100 + minutes

data['Zaman'] = data['Zaman'].apply(lambda x : 0 if x < 230 else ( 1 if (x > 230 and x < 400)  else  2))

"""
3rd step : Seperate X and y.
"""
X = pd.DataFrame(data.iloc[:,0:11].values)
y = pd.DataFrame(data.iloc[:,11].values)
X.columns = ['Zaman', 'Takim', 'Oyuncu', 'Puan', 'Dagilim', 'Zon', 'Kontrat', 'Renk', 'Kontr', 'Hedef', 'Sonuc']
"""
4st step : Convert Nominal data.
"""
X = pd.get_dummies(X, columns=['Takim', 'Oyuncu', 'Zon', 'Renk', 'Kontr', 'Hedef'])

"""
5st step : Normalize X
"""
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)

count = 27


"""
PCA Implementation
"""
variance_rat = np.zeros(count-1)
explained_count = 0;

for i in range(2,count-1):
    sklearn_pca = sklearnPCA(n_components=i)
    fit_results = sklearn_pca.fit(X)
    transform_results = sklearn_pca.transform(X)

    rat = np.sum(sklearn_pca.explained_variance_ratio_)
    print(rat)
    explained_count = i;
    variance_rat[i - 2] = rat;
    if rat >= 0.75:
        break

plt.bar(range(2,explained_count+1), variance_rat[0:explained_count-1], align='center', alpha=1)
plt.ylabel('Explained Variance')
plt.xlabel('Number of Attributes')
plt.title('Variance vs Number of Attribute')
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transform_results, y, test_size=0.3, random_state=0)

theta = np.zeros([1,explained_count])
theta, cost, test_cost = GradientDescent(X_train, y_train, theta, 0.001, 250, X_test, y_test)
print(theta)
print(ComputeCost(X_test, y_test, theta))

PlotCostValues(cost, test_cost)