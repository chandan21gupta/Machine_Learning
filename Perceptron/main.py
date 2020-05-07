import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self, learning_rate = 1, n_iter = 500, random_state = 1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        random_generator = np.random.RandomState(self.random_state)
        self.weights = random_generator.normal(loc = 0.0, scale = 0.1, size = 1+X.shape[1])
        self.errors = []

        for i in range(self.n_iter):
            print(self.errors)
            error = 0
            for features, target in zip(X,y):
                update = self.learning_rate*(target - self.predict(features))
                self.weights[1:] += update*features
                self.weights[0] += update
                error += int(update != 0.0)
                #print(update)
                #print(error)
            #print(error)
            self.errors.append(error)
        return self

    def predict(self, X):
        prediction = np.dot(X, self.weights[1:]) + self.weights[0]
        if(prediction >= 0):
            return 1
        return -1


df = pd.read_csv('Iris.csv')
#print(df.head())
temp = df.iloc[0:100,5].values
#print(temp)
y = []
for i in range (0,100):
    if str(temp[i]) == 'Iris-setosa':
        y.append(-1)
    else:
        y.append(1)
y = np.array(y)
X = df.iloc[0:100,[1,3]].values

ppn = perceptron(learning_rate=0.1, n_iter = 10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors) + 1),ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()



#print(y)
#print(X)
#print(df.head())
        


        