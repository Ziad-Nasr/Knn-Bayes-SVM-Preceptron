import samples as Sample
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np


def DataInput(ImageFileName,NumImages = 5000):
    FinalImage = []
    img = Sample.loadDataFile(ImageFileName,NumImages,28,28)
    for i in range(NumImages):
        FinalImage.append(np.array(img[i].getPixels()).flatten().tolist())
    return FinalImage

def Training(LabelFileName,Data):
    Label=Sample.loadLabelsFile(LabelFileName, 5000)
    GNB = GaussianNB()
    GNB.fit(Data,Label)
    return GNB

def Prediction(ImageFileName,GNB, NumImages = 1000):
    PredictionDataSet = []
    img = Sample.loadDataFile(ImageFileName,NumImages,28,28)
    for i in range(NumImages):
        PredictionDataSet.append(np.array(img[i].getPixels()).flatten().tolist())
    
    return GNB.predict(PredictionDataSet)
    
def CompareToReal(LabelFileName,PredictedDataSet, NumLabels = 1000):
    Wrong = 0
    Label=Sample.loadLabelsFile(LabelFileName, NumLabels)
    for i in range(NumLabels):
        if (PredictedDataSet[i] != Label[i]):
            Wrong += 1
    # print(PredictedDataSet)
    # print("Eshta yaba")
    # print(Label)
    return (100.0*(NumLabels - Wrong)/NumLabels)

# X, y = load_iris(return_X_y=True)
DataSet = DataInput("trainingimages")
BayesClassification = Training("traininglabels",DataSet) #Setting the classification with different K value in Euclidean Distance 
OutputPrediction = Prediction("validationimages",BayesClassification) #Prediction on another data set that is the Validation data. 
print(CompareToReal("validationlabels",OutputPrediction))
BayesClassification = Training("traininglabels",DataSet) #Setting the classification with different K value in Manhattan Distance 
OutputPrediction = Prediction("validationimages",BayesClassification) #Prediction on another data set that is the Validation data. 
print(CompareToReal("validationlabels",OutputPrediction))

# print("Number of mislabeled points out of a total %d points : %d"
    #   % (X_test.shape[0], (y_test != y_pred).sum()))