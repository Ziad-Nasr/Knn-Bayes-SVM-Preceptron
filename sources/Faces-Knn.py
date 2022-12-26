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
        print(i)
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
    print(NumImages)
    return KNN.predict(DataInput(ImageFileName,NumImages))

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

def VisualizingWrong(FileName,Items,Stuff,Num):
    img = Sample.loadDataFile(FileName,Num,28,28)
    Label=Sample.loadLabelsFile("testlabels", Num)
    for i in range(len(Items)):
        plt.subplot(3,4,i+1)
        plt.imshow(np.array(img[Items[i]].getPixels()).reshape(-1,1))
        plt.title(Stuff[Items[i]])

def VisualizingCorrect(FileName,Items,Stuff,Num):
    img = Sample.loadDataFile(FileName,Num,28,28)
    Label=Sample.loadLabelsFile("testlabels", Num)
    for i in range(len(Items)):
        plt.subplot(3,4,i+7)
        plt.imshow(img[Items[i]].getPixels())
        plt.title(Stuff[Items[i]])

DataSet = DataInput("facedatatrain")
TotalAccuracyEuclidean = []
TotalAccuracyManhattan = []
K = []
# for i in range(10):
#     print("Iteration %s for k = %s" % (i+1, i+1))
#     KnnClassification = Training("facedatatrainlabels",DataSet,i+1,Distance = Euclidean) #Setting the classification with different K value in Euclidean Distance 
#     OutputPrediction = Prediction("facedatavalidation",KnnClassification) #Prediction on another data set that is the Validation data. 
#     Accuracy=CompareToReal("facedatavalidationlabels",OutputPrediction)
#     print(Accuracy)
#     TotalAccuracyEuclidean.append(Accuracy)
#     KnnClassification = Training("facedatatrainlabels",DataSet,i+1,Distance = Manhattan) #Setting the classification with different K value in Manhattan Distance 
#     OutputPrediction = Prediction("facedatavalidation",KnnClassification) #Prediction on another data set that is the Validation data. 
#     Accuracy=CompareToReal("facedatavalidationlabels",OutputPrediction)
#     print(Accuracy)
#     TotalAccuracyManhattan.append(Accuracy)
#     K.append(i+1)

plt.plot(K,TotalAccuracyEuclidean,label="Euclidean")
plt.plot(K,TotalAccuracyManhattan,label="Manhattan")
plt.legend()
plt.show()


#Na2es Plotting el Graph ben el Manhattan w el Euclidean Distances
for i in range(2):
    print("Iteration %s for k = %s" % (i+1, i+1))
    KnnClassificationEuclidean = Training("facedatatrainlabels",DataSet,i+1,Distance = Euclidean) #Setting the classification with different K value in Euclidean Distance 
    OutputPredictionEuclidean = Prediction("facedatatest",KnnClassificationEuclidean, NumImages=150) #Prediction on another data set that is the Testing data. 
    Accuracy,SamplesWrongEuclidean,SamplesCorrectEuclidean = CompareToReal("facedatatrainlabels",OutputPredictionEuclidean,NumLabels = 150)
    print(Accuracy)
PlottingImg = []
print(SamplesWrongEuclidean,SamplesCorrectEuclidean)
for i in range(len(SamplesWrongEuclidean)):
    PlottingImg.append(SamplesWrongEuclidean[i])
VisualizingWrong('facedatatest',SamplesWrongEuclidean,OutputPredictionEuclidean,150)
VisualizingCorrect('facedatatest',SamplesCorrectEuclidean,OutputPredictionEuclidean,150)
plt.show()  