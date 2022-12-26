import samples as ld
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd

def img_to_feature(img):
        image = []
        for row in img:
            for pixel in row:
                image.append(pixel)
        return image

# p = 1 -> Manhattan Distance
# p = 2 -> Euclidean Distance
def train(k=5, p=1):
    IMG_NUM = 5000

    image_datums = ld.loadDataFile('trainingimages', IMG_NUM, 28,28)
    labels = ld.loadLabelsFile('traininglabels',IMG_NUM)

    images = []
    for i in range(IMG_NUM):
        images.append(img_to_feature(image_datums[i].getPixels()))

    knn = KNN(n_neighbors=k)
    knn.fit(images, labels)
    return knn

def testAccuracy(classifier, dataFile, labelFile, IMG_NUM):
    image_datums = ld.loadDataFile(dataFile, IMG_NUM, 28,28)
    labels = ld.loadLabelsFile(labelFile,IMG_NUM)

    count = 0

    for i in range(len(image_datums)):
        current = []
        current.append(img_to_feature(image_datums[i].getPixels()))
        prediction = classifier.predict(current)
        if prediction[0] == labels[i]:
            count += 1
    
    return count * 1.0 / IMG_NUM

def tuneHyperParams():
    out = []
    for i in range(30):
        print("Iteration:")
        print(i)
        knn = train(k=i+1, p=2)
        accuracy = testAccuracy(knn, "validationimages", "validationlabels", 1000)
        knn = train(k=i+1, p=1)
        accuracy_man = testAccuracy(knn, "validationimages", "validationlabels", 1000)
        out.append([i+1, accuracy, accuracy_man])
    print("Making CSV File...")
    df = pd.DataFrame(out)
    df.to_csv('knn.csv', index=False)

knn = train(k=1, p=1)
accuracy = testAccuracy(knn, "testimages", "testlabels", 1000)
print("Our Model's real accuracy")
print(accuracy)
    

    
