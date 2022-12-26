import samples as Sample
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

Manhattan = 1
Euclidean  = 2

#Responsible for Obtaining Data from training images and generating 1D array 
def DataInput(ImageFileName,NumImages = 5000):
    FinalImage = []
    img = Sample.loadDataFile(ImageFileName,NumImages,28,28)
    for i in range(NumImages):
        FinalImage.append(np.array(img[i].getPixels()).flatten().tolist())
    print(FinalImage,len(FinalImage))
    return FinalImage

def Training(LabelFileName,Data,knn,Distance = Manhattan,):
    Label=Sample.loadLabelsFile(LabelFileName, 5000)
    KNN = KNeighborsClassifier(n_neighbors=knn , p=Distance)
    KNN.fit(Data,Label)
    return KNN

def Prediction(ImageFileName,KNN, NumImages = 1000):
    PredictionDataSet = []
    img = Sample.loadDataFile(ImageFileName,NumImages,28,28)
    for i in range(NumImages):
        PredictionDataSet.append(np.array(img[i].getPixels()).flatten().tolist())
    return KNN.predict(PredictionDataSet)

def CompareToReal(LabelFileName,PredictedDataSet, NumLabels = 1000):
    Wrong = 0
    Label=Sample.loadLabelsFile(LabelFileName, NumLabels)
    for i in range(NumLabels):
        if (PredictedDataSet[i] != Label[i]):
            Wrong += 1
    return (100.0*(NumLabels - Wrong)/NumLabels)


DataSet = DataInput("trainingimages")
TotalAccuracyEuclidean = []
TotalAccuracyManhattan = []
K = []



for i in range(10):
    print("Iteration %s for k = %s" % (i+1, i+1))
    KnnClassification = Training("traininglabels",DataSet,i+1,Distance = Euclidean) #Setting the classification with different K value in Euclidean Distance 
    OutputPrediction = Prediction("validationimages",KnnClassification) #Prediction on another data set that is the Validation data. 
    Accuracy = CompareToReal("validationlabels",OutputPrediction)
    print(Accuracy)
    TotalAccuracyEuclidean.append(Accuracy)
    KnnClassification = Training("traininglabels",DataSet,i+1,Distance = Manhattan) #Setting the classification with different K value in Manhattan Distance 
    OutputPrediction = Prediction("validationimages",KnnClassification) #Prediction on another data set that is the Validation data. 
    Accuracy = CompareToReal("validationlabels",OutputPrediction)
    print(Accuracy)
    TotalAccuracyManhattan.append(Accuracy)
    K.append(i+1)
#Na2es Plotting el Graph ben el Manhattan w el Euclidean Distances
plt.plot(K,TotalAccuracyEuclidean,label="Euclidean")
plt.plot(K,TotalAccuracyManhattan,label="Manhattan")
plt.legend()
plt.show()
