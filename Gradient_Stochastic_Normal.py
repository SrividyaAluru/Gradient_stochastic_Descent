import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def getdata(filename):
    data = np.loadtxt(open("dataset.txt", "rb"), delimiter=",", skiprows=0)
    data = np.delete(data,0,axis=1)
    # data = ( data - np.mean(data, axis= 0) )/(np.std(data, axis= 0))
    data = ( data - np.min(data, axis= 0) )/(np.max(data, axis= 0)-np.min(data,axis= 0))

    # #print(data, [...])
    return data;
#data is in the numpy array

#have to split into train and test
def TrainTestSplit(data):
    train_size = int(data.shape[0]*80/100)
    test_size = int(data.shape[0]*20/100)
    #print(train_size,test_size, [...])
    train_data = data[0:train_size,:]
    #print(train_data,train_data.shape, [...])
    test_data = data[train_size:,:]
    #print(test_data,test_data.shape, [...])
    return train_data,test_data

def GetTrainPoints(data):
    return data[:,0],data[:,1],data[:,2]

def GetLossAndGradient(w0,w1,w2,x_train,y_train,z_train):
    #calculate sum of square of errors with the given weights
    #zcap=w0+w1*x_train+w2*y_train
    #E=0.5*sigma(zcap-z)
    #check if the operations lead to proper output
    zcap = w0 + x_train * w1 + y_train * w2
    #print('z_train ', z_train, z_train.shape)
    #print('zcap ', zcap,zcap.shape)
    err = zcap - z_train
    #print( np.sum( err**2 ) )
    #print('err ',np.sum(err),err.shape)
    #input()

    dw0 = np.sum(err)
    #print(dw0, [...])
    dw1 = np.sum(np.multiply(err,x_train))
    #print(dw1)
    dw2 = np.sum(np.multiply(err,y_train))
    #print(dw2)
    errsq = np.square(err)
    #print(errsq,errsq.shape)
    E = np.sum(errsq)
    #print(E, [...])
    return 0.5*E,dw0,dw1,dw2 #half sum of erors of all the points


#create func for finding gradiensts
#then epoch each time subtract eta time ths gradiensts
def GD(x_train,y_train,z_train,stochastic= False):
    #what is the stopping criterion
    w0,w1,w2 = np.random.normal(),np.random.normal(),np.random.normal()
    loss = []
      #learnign  rate
    if not stochastic:
        eta = 0.000001
        numberOfEpochs = 1000000
        prev_loss = float('inf')
        for x in range(numberOfEpochs):
            Eold,dw0,dw1,dw2 = GetLossAndGradient(w0,w1,w2,x_train,y_train,z_train)
            # print(x,":\t",Eold)
            if abs(Eold-prev_loss)<1e-7:
                break
            prev_loss = Eold
            if x%50==0:
                loss.append(Eold)
                print(Eold)
            #print('grad_w', dw0, dw1, dw2)
            w0 = w0 - eta * dw0
            w1 = w1 - eta * dw1
            w2 = w2 - eta * dw2

        return w0, w1, w2, loss
    else:
        numberOfEpochs = 5
        eta = 0.001
        for epoch in range(numberOfEpochs):
            for datapoint in zip(x_train,y_train,z_train):
                Eold,dw0,dw1,dw2 = GetLossAndGradient(w0,w1,w2,np.array([datapoint[0]]),np.array([datapoint[1]]),np.array([datapoint[2]]))
                loss.append(Eold)
                # print(Eold, [...])
                w0 = w0 - eta * dw0
                w1 = w1 - eta * dw1
                w2 = w2 - eta * dw2
            print(Eold,w0,w1,w2, [...])
        return w0,w1,w2,loss


def NormalEquation(data):
    x = data[:,:2]
    y = data[:,2]

    x = np.hstack((np.ones((x.shape[0],1)),x))

    w = np.linalg.inv((x.T).dot(x)).dot(x.T).dot(y)
    return w.tolist()

def GetTestLoss(w0,w1,w2,data):
    x_test,y_test,z_test = data[:,0],data[:,1],data[:,2]
    zcap = w0 + x_test * w1 + y_test * w2
    err = zcap - z_test
    errsq = np.sum(err**2)
    return 0.5*errsq

def GetFullError(data):
    z_test = data[:,2]
    deviation = z_test-np.mean(z_test)
    devsq = np.sum(deviation**2)
    return 0.5*devsq

def main():
    data = getdata("dataset.txt")
    #print(data,data.shape)
    #to randomly pick the train and test
    np.random.shuffle(data)
    #print(data,data.shape)
    train_data,test_data = TrainTestSplit(data)
    #save the training data and test data and use it for all the cases
    print('traindata',train_data)
    print('testdata',test_data)
    x_train,y_train,z_train = GetTrainPoints(train_data)
    #print(x_train,y_train,z_train)
    #once the poitns are there z_poitns is the true value

    w = NormalEquation(train_data)
    print('NormalEquation',w[0],w[1],w[2])
    test_loss = GetTestLoss(w[0],w[1],w[2],test_data)
    train_loss = GetTestLoss(w[0],w[1],w[2],train_data)
    print('Test Loss for this data ',test_loss)
    print('Train Loss for this data ',train_loss)
    E_mean = GetFullError(test_data)
    print("r^2:",test_loss/E_mean)

    # input()
    w0_ns,w1_ns,w2_ns,loss_ns = GD(x_train,y_train,z_train)
    print("non stochastic",w0_ns,w1_ns,w2_ns)
    test_loss = GetTestLoss(w0_ns,w1_ns,w2_ns,test_data)
    train_loss = GetTestLoss(w[0],w[1],w[2],train_data)
    print('Test Loss for this data ',test_loss)
    print('Train Loss for this data ',train_loss)
    E_mean = GetFullError(test_data)
    print("r^2:",test_loss/E_mean)


    plt.plot([i for i in range(len(loss_ns))],loss_ns)
    plt.show()

    w0_s,w1_s,w2_s,loss_s = GD(x_train,y_train,z_train,stochastic= True)
    print("stochastic",w0_s,w1_s,w2_s)

    test_loss = GetTestLoss(w0_s,w1_s,w2_s,test_data)
    train_loss = GetTestLoss(w0_s,w1_s,w2_s,train_data)
    print('Test Loss for this data ',test_loss)
    print('Train Loss for this data ',train_loss)
    E_mean = GetFullError(test_data)
    print("r^2:",test_loss/E_mean)

    plt.plot([i for i in range(len(loss_s))],loss_s)
    plt.show()




if __name__=="__main__":
    main()
