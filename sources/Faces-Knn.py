import samples as Sample
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

Manhattan = 1
Euclidean  = 2

def DataInput(ImageFileName,NumImages = 451):
    print(NumImages)
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

    visualize = np.array(flatter).reshape(NumImages,60, 70)      
    return flatter,visualize

def Training(LabelFileName,Data,knn,Distance = Manhattan,):
    Label=Sample.loadLabelsFile(LabelFileName, 451)
    KNN = KNeighborsClassifier(n_neighbors=knn , p=Distance)
    KNN.fit(Data,Label)
    return KNN

def Prediction(ImageFileName,KNN, NumImages = 301):
    PredictionDataSet = []
    # img = Sample.loadDataFile(ImageFileName,NumImages,60,70)
    # print(NumImages)
    return KNN.predict(DataInput(ImageFileName,NumImages)[0])

#Check if Predicted = actual then check Error compared to Total.
def CompareToReal(LabelFileName,PredictedDataSet, NumLabels = 301):
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

DataSet,vis = DataInput("facedatatrain")

def VisualizingWrong(FileName,Items,OutputItems,Num):
    img = Sample.loadDataFile(FileName,Num,60,70)
    # print(pr)
    print(OutputItems)
    print(Items)
    for i in range(len(Items)):
        plt.subplot(3,4,i+1)
        print("csbsdjkcbksdc")
        plt.imshow(vis[Items[i]])
        plt.title(OutputItems[Items[i]])

def VisualizingCorrect(FileName,Items,OutputItems,Num):
    img = Sample.loadDataFile(FileName,Num,60,70)
    print(OutputItems)
    Label=Sample.loadLabelsFile("testlabels", Num)
    for i in range(len(Items)):
        plt.subplot(3,4,i+7)
        plt.imshow(vis[Items[i]])
        plt.title(OutputItems[Items[i]])


TotalAccuracyEuclidean = []
TotalAccuracyManhattan = []
K = []
for i in range(10):
    print("Iteration %s for k = %s" % (i+1, i+1))
    KnnClassification = Training("facedatatrainlabels",DataSet,i+1,Distance = Euclidean) #Setting the classification with different K value in Euclidean Distance 
    OutputPrediction = Prediction("facedatavalidation",KnnClassification) #Prediction on another data set that is the Validation data. 
    Accuracy=CompareToReal("facedatavalidationlabels",OutputPrediction)[0]
    print(Accuracy)
    TotalAccuracyEuclidean.append(Accuracy)
    KnnClassification = Training("facedatatrainlabels",DataSet,i+1,Distance = Manhattan) #Setting the classification with different K value in Manhattan Distance 
    OutputPrediction = Prediction("facedatavalidation",KnnClassification) #Prediction on another data set that is the Validation data. 
    Accuracy=CompareToReal("facedatavalidationlabels",OutputPrediction)[0]
    print(Accuracy)
    TotalAccuracyManhattan.append(Accuracy)
    K.append(i+1)

plt.plot(K,TotalAccuracyEuclidean,label="Euclidean")
plt.plot(K,TotalAccuracyManhattan,label="Manhattan")
plt.legend()
plt.title("Valdiation K Values and Distance Comparison")
plt.show()
for i in range(5):
    print("Iteration %s for k = %s" % (i+1, i+1))
    KnnClassificationEuclidean = Training("facedatatrainlabels",DataSet,i+1,Distance = Euclidean) #Setting the classification with different K value in Euclidean Distance 
    OutputPredictionEuclidean = Prediction("facedatatest",KnnClassificationEuclidean, NumImages=150) #Prediction on another data set that is the Testing data. 
    Accuracy,SamplesWrongEuclidean,SamplesCorrectEuclidean = CompareToReal("facedatatrainlabels",OutputPredictionEuclidean,NumLabels = 150)
    print(Accuracy)
PlottingImg = []
for i in range(len(SamplesWrongEuclidean)):
    PlottingImg.append(SamplesWrongEuclidean[i])
VisualizingWrong('facedatatest',SamplesWrongEuclidean,OutputPredictionEuclidean,150)
VisualizingCorrect('facedatatest',SamplesCorrectEuclidean,OutputPredictionEuclidean,150)
plt.show()  