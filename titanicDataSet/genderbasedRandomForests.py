from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")

train_data = dataset[0::,1].values
test_data = dataset[0::,1]

#Random forest only takes in numbers
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data[0::,1::],train_data[0::,0])

output = forest.predict(test_data)
