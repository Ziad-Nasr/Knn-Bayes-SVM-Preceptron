import samples as Sample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

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

    visualize = np.array(flatter).reshape(NumImages,60, 70)      
    return flatter,visualize

def Training(LabelFileName,Data):
    Label=Sample.loadLabelsFile(LabelFileName, 451)
    GNB = GaussianNB()
    GNB.fit(Data,Label)
    return GNB

def Prediction(ImageFileName,GNB, NumImages = 301):
    return GNB.predict(DataInput(ImageFileName,NumImages)[0])

#Check if Predicted = actual then check Error compared to Total.
def CompareToReal(LabelFileName,PredictedDataSet, NumLabels = 300):
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
    Faces = DataInput("facedatatest", NumImages = Num)[1]
    for i in range(len(Items)):
        plt.subplot(3,4,i+1)
        plt.imshow(Faces[Items[i]])
        plt.title(OutputItems[Items[i]])

def VisualizingCorrect(FileName,Items,OutputItems,Num):
    Faces = DataInput("facedatatest",NumImages=Num)[1]
    for i in range(len(Items)):
        plt.subplot(3,4,i+7)
        plt.imshow(Faces[Items[i]])
        plt.title(OutputItems[Items[i]])


BayesClassification = Training("facedatatrainlabels",DataSet) #Setting the classification.
OutputPrediction = Prediction("facedatavalidation",BayesClassification) #Prediction on another data set that is the Validation data. 
Accuracy,SamplesWrong,SamplesCorrect =CompareToReal("facedatavalidationlabels",OutputPrediction)
print(Accuracy)

BayesClassification = Training("facedatatrainlabels",DataSet) #Setting the classification.
OutputPrediction = Prediction("facedatatest",BayesClassification, NumImages=150) #Prediction on another data set that is the Validation data. 
Accuracy,SamplesWrong,SamplesCorrect =CompareToReal("facedatatestlabels",OutputPrediction,NumLabels = 150)
print(Accuracy)

VisualizingWrong('facedatatest',SamplesWrong,OutputPrediction,150)
VisualizingCorrect('facedatatest',SamplesCorrect,OutputPrediction,150)
plt.show()  

print("The Accuracy of the Test Data is = %s" % (Accuracy))

