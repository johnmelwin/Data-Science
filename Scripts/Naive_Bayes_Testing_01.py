from sklearn.naive_bayes import GaussianNB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load dataset
    data_train = pd.read_csv(r"C:\Users\jmelw\Desktop\FALL 22-23\FOUNDATIONS OF DS\Data-Science\data\Iris_train.csv")
    # plot dataset

    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.scatterplot(data=data_train, x='SepalLengthCm', y="PetalLengthCm", hue="Species")
    plt.show()
    plt.close()

    # seperate dependent and independent variables
    independent_variables = ["SepalLengthCm", "PetalLengthCm", "PetalWidthCm", "SepalLengthCm"]
    dependent_variable = ["Species"]
    # assign X and Y
    X = data_train[independent_variables]
    Y = data_train["Species"]
    #train
    clf = GaussianNB()
    clf.fit(X,Y)
    #setting test data
    data_test = pd.read_csv(r"C:\Users\jmelw\Desktop\FALL 22-23\FOUNDATIONS OF DS\Data-Science\data\Iris_test.csv")
    X_test = data_test[independent_variables]
    #test(predict)
    predictions = clf.predict(X_test)
    print(predictions)

    # Predict probabilities
    probs = clf.predict_proba(X_test)
    print("probs \n" + str(probs))
    #print results
    for i,pred in enumerate(predictions):
        print("%s\t%f" %(pred,max(probs[i])))
