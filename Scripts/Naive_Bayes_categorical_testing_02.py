from sklearn.naive_bayes import CategoricalNB
import pandas as pd

if __name__ == "__main__":
    #load data
    data = pd.read_csv(r"C:\Users\jmelw\Desktop\FALL 22-23\FOUNDATIONS OF DS\Data-Science\data\class_train.csv")
    #convert categorical data to numerical
    print(data)
    data["Attrib1"].replace(["Yes", "No"],[1,0], inplace=True)
    data["Attrib2"].replace(["Large", "Medium", "Small"], [3,2,1 ], inplace=True)
    print("after modification")
    print(data)

    #seperate dependent and independent data
    dependent_variable = ["Attrib1", "Attrib2", "Attrib3"]
    X = data[dependent_variable]
    Y = data["Class"]
    #train
    clf = CategoricalNB()
    clf.fit(X,Y)
    #load test
    data_test = pd.read_csv(r"C:\Users\jmelw\Desktop\FALL 22-23\FOUNDATIONS OF DS\Data-Science\data\class_test.csv")
    data_test["Attrib1"].replace(["Yes", "No"], [1, 0], inplace=True)
    data_test["Attrib2"].replace(["Large", "Medium", "Small"], [3, 2, 1], inplace=True)
    X_test = data_test[dependent_variable]
    print("X_test")
    print(X_test)

    #predict'
    predictions = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    print("probs")
    print(probs)
    print("predictions")
    print(predictions)

    for i,pred in enumerate(predictions):
        print("%s\t%f" %(pred,max(probs[i])))
