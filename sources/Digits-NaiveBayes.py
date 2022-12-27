import samples as Sample
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt


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
    indecesWrong = []
    indecesCorrect = []
    Label=Sample.loadLabelsFile(LabelFileName, NumLabels)
    for i in range(NumLabels):
        if (PredictedDataSet[i] != Label[i]):
            Wrong += 1
            if (len(indecesWrong) < 6):
                indecesWrong.append(i)
        elif (len(indecesCorrect) < 6):
            indecesCorrect.append(i)
    return (100.0*(NumLabels - Wrong)/NumLabels),indecesWrong,indecesCorrect

def VisualizingWrong(FileName,Items,Stuff):
    img = Sample.loadDataFile(FileName,1000,28,28)
    for i in range(len(Items)):
        plt.subplot(3,4,i+1)
        plt.imshow(np.array(img[Items[i]].getPixels()))
        plt.title(Stuff[Items[i]])

def VisualizingCorrect(FileName,Items,Stuff):
    img = Sample.loadDataFile(FileName,1000,28,28)
    for i in range(len(Items)):
        plt.subplot(3,4,i+7)
        plt.imshow(img[Items[i]].getPixels())
        plt.title(Stuff[Items[i]])

DataSet = DataInput("trainingimages")
BayesClassification = Training("traininglabels",DataSet) #Setting the classification. 
OutputPrediction = Prediction("validationimages",BayesClassification) #Prediction on another data set that is the Validation data. 
Accuracy = CompareToReal("validationlabels",OutputPrediction)[0]
print(Accuracy)

BayesClassification = Training("traininglabels",DataSet) #Setting the classification.
OutputPrediction = Prediction("testimages",BayesClassification) #Prediction on another data set that is the Validation data. 
Accuracy,SamplesWrongEuclidean,SamplesCorrectEuclidean = CompareToReal("testlabels",OutputPrediction)
print(Accuracy)

PlottingImg = []
for i in range(len(SamplesWrongEuclidean)):
    PlottingImg.append(SamplesWrongEuclidean[i])
VisualizingWrong('testimages',SamplesWrongEuclidean,OutputPrediction)
VisualizingCorrect('testimages',SamplesCorrectEuclidean,OutputPrediction)
plt.show()

print("The Accuracy is = %s" % (Accuracy))
