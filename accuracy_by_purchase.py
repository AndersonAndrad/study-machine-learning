import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

data = pd.read_csv(uri)

#features
x = data[["home", "how_it_works", "contact"]]
y = data[["bought"]]

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

seed = 20

train_x, test_x, train_y, test_y = train_test_split(x,y, random_state= seed, test_size=0.25, stratify=y)

model = LinearSVC()
model.fit(train_x, train_y)

forecast = model.predict(test_x)

accuracy = accuracy_score(test_y, forecast) * 100

print('accuracy is: ', accuracy)
