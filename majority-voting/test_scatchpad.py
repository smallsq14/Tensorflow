import numpy as np


pred1 = np.load("all_predictions_0_.txt.npy")
pred2 = np.load("all_predictions_0_.txt.npy")
pred3 = np.load("all_predictions_0_.txt.npy")
testList = list()
testList.append(pred1)
testList.append(pred2)
testList.append(pred3)
number_of_classifiers = 3
pos_value = np.array([0, 1])
neg_value = np.array([1, 0])
all_predictions = []
for w in range(0, 1000):
    sumOne = 0
    sumZero = 0
    for t in range(0, number_of_classifiers):
       # print("classifier {} prediction: {}".format(t, testList[t][w]))
        if (testList[t][w] == 1.0).all():
            #print("Positive Label")
            sumOne = sumOne + 1
        else:
           # print("Negative label")
            sumZero = sumZero + 1
    if (sumOne > sumZero):
        all_predictions.append(1.0)
    else:
        all_predictions.append(0.0)
    if(sumOne==2):
        print("condition voted")
    if (sumZero == 2):
        print("condition voted")
    #testagain = np.argmax(np.array(all_predictions).astype(float), axis=1)
    testagain = np.array(all_predictions)
    np.array_equal(testagain,testList[0])

print ("end")
