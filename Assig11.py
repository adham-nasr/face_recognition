import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import transpose
from numpy.linalg import eig
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

def PCA(Mat,alpha):
    center_Mat = Mat-Mat.mean(0)
    cov_Mat = np.cov(center_Mat , rowvar=False)
    eigVal,eigVec = np.linalg.eigh(cov_Mat)

    den = np.sum(eigVal);
    num = 0.0;
    r = 0;
    ind_sort = np.argsort(eigVal)[::-1]
    for i in range(np.size(eigVal)):
        r = r+1;
        num += ind_sort[i]
        if num / den >= alpha:
            break;
    UD = eigVec[:,ind_sort]
    UR = UD[:,:r];
    return UR


def Reduce(Proj_Mat,X):
  center_Mat = X-X.mean(0)
  return  np.transpose(np.dot(np.transpose(Proj_Mat),np.transpose(center_Mat)))


def Learn(D_train,D_test,y_train,y_test,label):
  Proj_Mat = [0,0,0,0]
  Proj_Mat[0] = PCA(D_train,0.8);
  Proj_Mat[1] = PCA(D_train,0.85);
  Proj_Mat[2] = PCA(D_train,0.9);
  Proj_Mat[3] = PCA(D_train,0.95);

  Red_train = [0,0,0,0]
  Red_test = [0,0,0,0]

  for i in range(4):
    Red_train[i] = Reduce(Proj_Mat[i],D_train)
    Red_test[i]  = Reduce(Proj_Mat[i],D_test)

  # CLASSIFICATION
  knn = KNeighborsClassifier(n_neighbors=1)
  PLT_X = [0.8,0.85,0.9,0.95]
  PLT_y = []
  for i in range(4):
    knn.fit(Red_train[i],y_train)
    #
    predictions = knn.predict(Red_test[i])

    #print("predictions")
    #print(predictions)
    #for j in range(len(predictions)):

    #print("-------------")
    #
    print("accuracy")
    accuracy = knn.score(Red_test[i],y_test);
    print(accuracy)
    print("-----------")
    PLT_y.append(accuracy)
  l = [PLT_X,PLT_y,label]
  #plt.plot(PLT_X,PLT_y,label=label)
  #plt.plot(PLT_X,PLT_y,'bo')
  #plt.legend()
  #plt.xlabel('Alpha values')
  #plt.ylabel('Accuracy')
  #plt.show()

  ### CLassifier Tuning
  PLT_X = []
  PLT_y = []
  for k in range(1,10,2): 
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy=0;
    for i in range(4):
      knn.fit(Red_train[i],y_train)
      #
      predictions = knn.predict(Red_test[i])

      #print("predictions")
      #print(predictions)
      #for j in range(len(predictions)):

      #print("-------------")
      #
      print("accuracy")
      accuracy = knn.score(Red_test[i],y_test);
      print(accuracy)
      print("-----------")
    PLT_X.append(k);
    PLT_y.append(accuracy);

  l2 = [PLT_X,PLT_y,label]
  #plt.plot(PLT_X,PLT_y,label=label)
  #plt.legend()
  #plt.xlabel('K values')
  #plt.ylabel('Accuracy')
  #plt.plot(1,PLT_y[0],'bo')
  #plt.show()

  del Proj_Mat
  del Red_train
  del Red_test

  return [l,l2]


#### MAIN

l = []
l2= []

for i in range(1,41):
    for j in range(1,11):
        fol = str(i)
        fil = str(j)
        img = Image.open(f"/content/gdrive/MyDrive/att_faces/s{fol}/{fil}.pgm")
        vec_img = np.array(img).flatten()
        l.append(vec_img)
    
D = np.array(l);

for i in range(1,41):
    for j in range(1,11):
        l2.append(i);

y = np.array(l2);

#50%
D_train1 = D[::2]
D_test1 = D[1::2]
y_train1 = y[::2]
y_test1 = y[1::2]

#70%
selector1=[]
selector2=[]
for i in range(400):
  if i%10 < 7:
    selector1.append(i)
  else:
    selector2.append(i)
D_train2 = D[np.array(selector1)]
D_test2 = D[np.array(selector2)]
y_train2 = y[np.array(selector1)]
y_test2 = y[np.array(selector2)]

del l
del l2
del D
del y

Lab1 = "50-50"
Lab2 = "70-30"

Plt_Data1 = Learn(D_train1,D_test1,y_train1,y_test1,Lab1)
Plt_Data2 = Learn(D_train2,D_test2,y_train2,y_test2,Lab2)



xlabel = ['Alpha values K=1','K values']
ylabel = ['Accuracy','Accuracy']

for i in range(2):
  PLT_X,PLT_y,label = Plt_Data1[i][0] , Plt_Data1[i][1] , Plt_Data1[i][2]
  plt.plot(PLT_X,PLT_y,color='g',label=label)
  plt.plot(PLT_X,PLT_y,'go')
  if i==0:
    plt.ylim(0.85,0.95)
  PLT_X,PLT_y,label = Plt_Data2[i][0] , Plt_Data2[i][1] , Plt_Data2[i][2]
  plt.plot(PLT_X,PLT_y,color='b',label=label)
  plt.plot(PLT_X,PLT_y,'bo')
  plt.legend()
  plt.xlabel(xlabel[i])
  plt.ylabel(ylabel[i])
  plt.show()

