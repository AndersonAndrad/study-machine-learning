#features
# [long hair, short leg, do auau] false = 0; true = 1
first_pig = [0,1,0]
second_pig = [0,1,1]
third_pig = [1,1,0]

first_dog = [0,1,1]
second_dog = [1,0,1]
third_dog = [1,1,1]

data = [first_pig, second_pig, third_pig, first_dog, second_dog, third_dog]

allClass = [1,1,1,0,0,0]

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
model.fit(data, allClass)

first_random_animal = [1,1,1]
second_random_animal = [1,1,0]
third_random_animal = [0,1,1]

base_test = [first_random_animal, second_random_animal, third_random_animal]

forecast = model.predict(base_test)

test_classes = [0,1,1]

accurary = accuracy_score (test_classes, forecast)

print('accurary is: ', accurary * 100)
