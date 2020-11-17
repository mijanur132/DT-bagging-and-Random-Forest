import os
import struct
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from trees import*
from plots import *
from scipy import stats
from statistics import *

num_folds = 10
bagging_num=30

def prepareData(trainSet):
    trainSet = trainSet.sample(random_state = 18, frac = 1).reset_index(drop = True)
    trainingSet = trainSet.sample(random_state = 32, frac = 0.5).reset_index(drop = True)
    fullTrainSetSize=trainingSet.shape[0]
    Foldsize = int(0.1 * fullTrainSetSize)

    fold1 = []
    for i in range(num_folds):
         fold1.append(trainingSet.values[(i)*Foldsize:(i+1)*Foldsize])
    return trainingSet,fold1

def stderr_avg(bg,depths,num_folds):
    bg_stderr = []
    bg_avgacc = []
    for num_tree in depths:
        sigma_l = stdev(bg[num_tree])
        bg_stderr.append(sigma_l / np.sqrt(num_folds))
        bg_avgacc.append(np.mean(bg[num_tree]))

    return bg_stderr,bg_avgacc





confusion_matrix = np.zeros((10, 10))  # needed to calculate confusion matrix
def build_confusion_matrix(predicted_value,actual_value):
    global confusion_matrix
    confusion_matrix[actual_value][predicted_value]+=1

def calculate_f1_score(confusion_matrix):
    true_pos = np.zeros(10)
    false_pos = np.zeros(10)
    false_neg = np.zeros(10)
    f1_accuracy=  np.zeros(10)
    precision=np.zeros(10)
    recall=np.zeros(10)
    count_label_present=0
    countp=10
    countr=10
    precision_avg=0
    recall_avg=0

    for conf_matrix_ind in range(0, 10):
        true_pos[conf_matrix_ind] = confusion_matrix[conf_matrix_ind][conf_matrix_ind]
        false_pos[conf_matrix_ind] = sum(row[conf_matrix_ind] for row in confusion_matrix) - true_pos[
            conf_matrix_ind]  # sum of column at confmatrixInd
        false_neg[conf_matrix_ind] = sum(confusion_matrix[conf_matrix_ind]) - true_pos[
            conf_matrix_ind]  # sum of row at confMatrixInd
        if (true_pos[conf_matrix_ind]+false_pos[conf_matrix_ind] + false_neg[conf_matrix_ind])!=0:
            f1_accuracy[conf_matrix_ind]=2 * true_pos[conf_matrix_ind] /(2*true_pos[conf_matrix_ind]+false_pos[conf_matrix_ind] + false_neg[conf_matrix_ind]) * 100
            count_label_present+=1
        if (true_pos[conf_matrix_ind]+false_pos[conf_matrix_ind])>0.1:
            precision[conf_matrix_ind]=true_pos[conf_matrix_ind]/(true_pos[conf_matrix_ind]+false_pos[conf_matrix_ind])
        else:
            countp-=1
        if (true_pos[conf_matrix_ind] + false_neg[conf_matrix_ind]) > 0.1:
            recall[conf_matrix_ind] = true_pos[conf_matrix_ind] / (true_pos[conf_matrix_ind] + false_neg[conf_matrix_ind])
        else:
            countr-=1
    precision_avg = sum(precision) / (countp)
    recall_avg = sum(recall) / (countr)
    f1_accuracy_sid=2*(precision_avg*recall_avg)/(precision_avg+recall_avg)*100
    sum_diag = np.trace(np.asanyarray(confusion_matrix))  # gives sum of all the diags
    test_accuracy = 100 * sum_diag / np.sum(np.asarray(confusion_matrix))  # sum_diag/sum_off_total_matrix

    return((test_accuracy,f1_accuracy_sid))


def read_data(data_set):
    if data_set is "training":
        image = 'train-images.idx3-ubyte'
        label = 'train-labels.idx1-ubyte'
    elif data_set is "testing":
        image = 't10k-images.idx3-ubyte'
        label = 't10k-labels.idx1-ubyte'
    else:
        print("not 'testing' or 'training'")

    with open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])
    for i in range(len(lbl)):
        yield get_img(i)

def data_process(name,lenz):

    train_set=np.zeros((lenz,786))
    training_data=list(read_data(name))
    for i in range(lenz):
        label,pixels=training_data[i]
        pixels=pixels.flatten()

        for j in range(len(pixels)):
            train_set[i][j] = float(pixels[j])/float(255)
        train_set[i][784] = 1                                      # add bias to train_set[784]
        train_set[i][785]=int(label)                               #add label to train_set[785]
    return train_set

def dot(w,item):
    d=np.dot(w,np.transpose(item))
    return d

def summ(a,b):
    for i in range(len(a)):
        a[i]=a[i]+b[i]
    return a

def read_perceptron(w,data):
    #print(w.shape)
    #print("ds",data.shape)
    for item in data:
        y = int(item[-1])  # label
        item_attr=item[:-1]
        o = np.argmax(dot(w, item_attr))
        build_confusion_matrix(o,y)



#seems ok
def batch_grad_dec(train_set,w,max_iter,c):
    for i in range(max_iter):
        del_w = np.zeros((10,784))
        #del_w=del_w+0.1*w
        for item in train_set:
            y=int(item[785] ) #label
            b=item[784]   #bias
            item_attr=item[:-2]             #remove last 2 entry as they are not attribute
            o=dot(w,item_attr)
            for ib in range(len(w)):
                oib=1/(1+np.exp(-o[ib]))
                #print(o[ib],oib)
                if ib==y:
                    y1=1
                    for ia in range(len(w[1])):
                        del_w[ib][ia]-=c*(oib-y1)*item_attr[ia]
                else:
                    y1=0
                    for ia in range(len(w[1])):
                        del_w[ib][ia] -= c * (oib - y1) * item_attr[ia]

        w+=del_w
    abs_w=np.linalg.norm((w), ord=1)
    print(abs_w)
    return w


#best upto now:
def batch_grad_dec_f1(train_set,w,LR):
    data_dim=len(w[0])
    classes_N=2 #number of classes
    #del_w = np.zeros((classes_N, data_dim))
    #print(w[0].size, del_w[0].size)

    #del_w=del_w+0.1*np.linalg.norm(w, ord=1)
    #del_w1 = np.zeros((classes_N, data_dim))
    for item in train_set:
        #print("item",item)
        del_w = np.zeros((classes_N, data_dim))
        oib=np.zeros(classes_N)
        y1=np.zeros(classes_N)
        y=int(item[len(train_set[0])-1] ) #label add label here
        #print("y",y)
        b=item[len(train_set[0])-2]   #bias
        #print("b",b)
        #print(item.size)
        item=item[:-1]             #remove last 2 entry as they are not attribute
        #print("updated item",item)
        #print("updated item size:",item.size, item[0].size)
        oib=dot(w,item)
        oib=1/(1+np.exp(-oib))
        y1[y]=1
        y2=LR*(oib-y1)
        for ib in range(len(del_w)):
            del_w[ib] -= y2[ib]*item
        w+=del_w
       # del_w1+= del_w
    err_or = np.zeros(classes_N)
    for item in train_set:
        y=int(item[-1])
        #print(y)
        item = item[:-1]
        #item_attr = item[:-2]
        g_wx = dot(w, item)
        g_wx = 1 / (1 + np.exp(-g_wx))
        M=np.log(g_wx)
        N=np.log(1-g_wx)
        y11 = np.zeros(classes_N)  #predicted level modified to 0 and 1
        #y = int(item[data_dim+1])  # label
        y11[y] = 1
        for ib in range(len(del_w)):
            err_or[ib] -= (y11[ib]* M[ib]+(1-y11[ib])*N[ib])
    err_or=sum(err_or)/classes_N
    return ((w,err_or))


def perceptron_func(train_set,test_set,trY,tsY,w,learningR):
    train_set,test_set=np.array(train_set),np.array(test_set)
    #print(train_set)
    data_dim = len(train_set[0])+2
    train_setL=len(train_set)
    test_setL = len(test_set)
    #print("trL",train_setL)
    #print("data",data_dim)
    #print(trY.shape)
    train_setwithBiasLabel=np.random.random((train_setL,data_dim))
    train_setwithBiasLabel[:,data_dim-1]=np.array(trY)[:]
    train_setwithBiasLabel[:, data_dim-2] = 1
    train_setwithBiasLabel[:,0:data_dim-2]=train_set[:,:]
    #print("trainsetwithbL",train_setwithBiasLabel.size, len(train_setwithBiasLabel[0]))

    tst_setwithBiasLabel=np.random.random((test_setL,data_dim))
    #print(tst_setwithBiasLabel.shape,tsY.shape)
    tst_setwithBiasLabel[:,data_dim-1]=np.array(tsY)[:]
    tst_setwithBiasLabel[:, data_dim-2] = 1
    tst_setwithBiasLabel[:, 0:data_dim - 2] = test_set[:, :]

    classes_N=2


    global confusion_matrix
    confusion_matrix = np.zeros((10, 10))

    wx = batch_grad_dec_f1(train_setwithBiasLabel, w,learningR)
    w = wx[0]
    errr = wx[1] * 100 / 10000
    #print(errr)
    read_perceptron(w, train_setwithBiasLabel)
    result = calculate_f1_score(confusion_matrix)
    confusion_matrix = np.zeros((10, 10))
    read_perceptron(w, tst_setwithBiasLabel)
    resultl = calculate_f1_score(confusion_matrix)

    fold_acc=resultl[0] / 100
    return w, fold_acc

def run_model(trees,fold,trainingSet):
    columns = trainingSet.columns
    models =  ["perceptron"]
    iteranationN = 10
    dt,bg,rf = {},{},{}
    fold_accN=[]
    classes_N = 2
    data_dim = len(fold[0][0]) + 1
    w = np.random.random((classes_N, data_dim - 1))
    c_s=[0.1,0.01,0.001,0.0001,0.00001]
    csList=list(range(1,len(c_s)))
    print(csList)
    for c in c_s:
        print("c",c)
        for iterI in range(iteranationN):
            fold_acc = []
            for i in range(num_folds):
                #print("fold:",i)
                testData = fold[i]
                folds=list(range(num_folds))
                del folds[i]
                trainData=np.array(fold[folds[0]])
                del folds[0]
                for j in folds:
                    trainData = np.vstack((trainData, np.array(fold[j])))
                trainData=np.array((trainData))
                trainData=trainData.reshape((-1,trainingSet.shape[1]))
                tdata, tstdata = pd.DataFrame(trainData, columns=columns), pd.DataFrame(testData, columns=columns)
                tdata_ylabel=tdata["decision"]
                tstdata_ylabel = tstdata["decision"]
                #print(tdata_ylabel,tstdata_ylabel)
                tdata=tdata.drop("decision",axis=1)
                tstdata=tstdata.drop("decision",axis=1)
                w,fold_acc1=perceptron_func(tdata,tstdata,tdata_ylabel,tstdata_ylabel,w,c)
                if iterI==iteranationN-1:
                    fold_acc.append(fold_acc1)
        fold_accN.append(np.mean(fold_acc))
        print(fold_acc,np.mean(fold_acc))
    print("fold acc", fold_acc, np.mean(fold_acc))
    print("fold accN", fold_accN, np.mean(fold_accN))
    #dt_stderr, dt_avgacc = stderr_avg(fold_accN, range(5), num_folds)
    css=["0.1","0.01","0.001","0.0001","0.00001"]
    cv_perceptron_plotLR(css, fold_accN)



def main():
    print("nothing")
    trees = __import__('trees')
    trainSet = pd.read_csv("trainingSet.csv")
    trainingSet, fold = prepareData(trainSet)
    run_model(trees, fold, trainingSet)



if __name__ == "__main__":
    main()