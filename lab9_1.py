import pandas as pd
import numpy as np

# Búsqueda por similitud en Iris

iris = pd.read_csv("iris.data", header=None)
data = iris.iloc[:, :4]

# Eliminar los objetos de consulta de data

data.drop([15, 82, 121], axis=0, inplace=True)
print(data.shape)

# Distancia euclediana ED
ED = lambda X, Y: (sum((X - Y) ** 2)) ** 0.5


# RANGE SEARCH

def rangeSearch(Q, r, data):
    result = []
    for id_, element in data.iterrows():
        dist = ED(Q, element)
        if dist <= r:
            result.append((id_, dist))
    return result


print(rangeSearch(data.iloc[1], 2, data))
ndf = pd.DataFrame(columns=['q15', 'q82', 'q121'])

for ind in [15, 82, 121]:
    for r in [0.8, 1.5, 2.6]:
        Q = iris.iloc[ind, :4]
        result = rangeSearch(Q, r, data)
        # precision
        target = iris.iloc[ind, 4]
        pr = 0
        for (id_, dist) in result:
            if target == iris.iloc[id_, 4]:
                pr += 1
        pr = pr / len(result)
        print("Q" + str(ind) + ": r-", r, " :pr-", pr)


# KNN SEARCH

def knnSearch(Q, k, data):
    result = []
    for id_, element in data.iterrows():
        dist = ED(Q, element)
        if dist <= r:
            result.append((id_, dist))
    result_ord = sorted(result, key=lambda x: x[1])
    return result_ord[:k]


for ind in [15, 82, 121]:
    for k in [2, 4, 8, 16, 32]:
        Q = iris.iloc[ind, :4]
        result = knnSearch(Q, k, data)
        # precision
        target = iris.iloc[ind, 4]
        pr = 0
        for (id_, dist) in result:
            if target == iris.iloc[id_, 4]:
                pr += 1
        pr = pr / len(result)
        print("Q" + str(ind) + ": k-", k, " :pr-", pr)
