import samples as Sample
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

Manhattan = 1
Euclidean  = 2

def DataInput(ImageFileName,NumImages = 451):
    FinalImage = []
    img = Sample.loadDataFile(ImageFileName,NumImages,60,70)
    for i in range(NumImages):
        FinalImage.append(np.array(img[i].getPixels()).flatten().tolist())
    updated_array=[]
    for i in range(NumImages):
        templist=[]
        for j in range(len(FinalImage[i])):
            if len(FinalImage[i][j]) != 0:
                templist.append(FinalImage[i][j])
        updated_array.append(templist)

    tempdata = np.array(updated_array)
    flatter =[]
    for i in range(NumImages):
        flatter.append(tempdata[i].flatten().tolist())   
    return flatter

def Training(LabelFileName,Data,knn,Distance = Manhattan,):
    Label=Sample.loadLabelsFile(LabelFileName, 451)
    KNN = KNeighborsClassifier(n_neighbors=knn , p=Distance)
    KNN.fit(Data,Label)
    return KNN

def Prediction(ImageFileName,KNN, NumImages = 301):
    PredictionDataSet = []
    img = Sample.loadDataFile(ImageFileName,NumImages,60,70)
    return KNN.predict(DataInput(ImageFileName,NumImages))

#Check if Predicted = actual then check Error compared to Total.
def CompareToReal(LabelFileName,PredictedDataSet, NumLabels = 301):
    Wrong = 0
    Label=Sample.loadLabelsFile(LabelFileName, NumLabels)
    for i in range(NumLabels):
        if (PredictedDataSet[i] != Label[i]):
            Wrong += 1
    return (100.0*(NumLabels - Wrong)/NumLabels)

DataSet = DataInput("facedatatrain")
TotalAccuracyEuclidean = []
TotalAccuracyManhattan = []
K = []
for i in range(10):
    print("Iteration %s for k = %s" % (i+1, i+1))
    KnnClassification = Training("facedatatrainlabels",DataSet,i+1,Distance = Euclidean) #Setting the classification with different K value in Euclidean Distance 
    OutputPrediction = Prediction("facedatavalidation",KnnClassification) #Prediction on another data set that is the Validation data. 
    Accuracy=CompareToReal("facedatavalidationlabels",OutputPrediction)
    print(Accuracy)
    TotalAccuracyEuclidean.append(Accuracy)
    KnnClassification = Training("facedatatrainlabels",DataSet,i+1,Distance = Manhattan) #Setting the classification with different K value in Manhattan Distance 
    OutputPrediction = Prediction("facedatavalidation",KnnClassification) #Prediction on another data set that is the Validation data. 
    Accuracy=CompareToReal("facedatavalidationlabels",OutputPrediction)
    print(Accuracy)
    TotalAccuracyManhattan.append(Accuracy)
    K.append(i+1)

plt.plot(K,TotalAccuracyEuclidean,label="Euclidean")
plt.plot(K,TotalAccuracyManhattan,label="Manhattan")
plt.legend()
plt.show()


#Na2es Plotting el Graph ben el Manhattan w el Euclidean Distances
