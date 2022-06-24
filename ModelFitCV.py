from sklearn.metrics import confusion_matrix, recall_score, f1_score, mean_absolute_error
import numpy as np # arrays, fourier transformatie, matrixen, domein lineaire algabra
from statistics import mean, stdev
    
def ModelTrainCV(model, skf, x_train, y_train):
    y_train_list = y_train.iloc()
    spec_list=[]
    sens_list=[]
    f1_list=[]
    accuracy_list=[]

    for train_index, test_index in skf.split(x_train,y_train):
        #print("Train:", train_index, "Test:", test_index)
        X_train_CV, X_test_CV = x_train[train_index], x_train[test_index]
        Y_train_CV, Y_test_CV = y_train_list[train_index], y_train_list[test_index]
        
        model.fit(X_train_CV, Y_train_CV)
        predict_Y_test_CV = model.predict(X_test_CV)

        f1 = f1_score(Y_test_CV,predict_Y_test_CV, pos_label=0)
        f1_list.append(f1)

        spec = recall_score(Y_test_CV,predict_Y_test_CV,pos_label=0)
        spec_list.append(spec)
    
        sens = recall_score(Y_test_CV,predict_Y_test_CV,pos_label=1)
        sens_list.append(sens)

        accuracy = model.score(X_test_CV,Y_test_CV)
        accuracy_list.append(accuracy)

    return f1_list, accuracy_list, spec_list, sens_list, model
   
def PrintTrainCVScores(f1_list, accuracy_list, spec_list, sens_list):
    PrintListScoresTrainCV('f1', f1_list, False)
    PrintListScoresTrainCV('specificity', spec_list, True)
    PrintListScoresTrainCV('accuracy', accuracy_list, True)
    PrintListScoresTrainCV('sensitivity', sens_list, True)

def PrintListScoresTrainCV(listType, list, showAsPercentages):
    if(showAsPercentages):
        max, min, avg = CalculatePercentages(list)  
    else:
       max, min, avg = CalculateDeimcals(list)

    print(f'List of possible {listType}:', list)
    print(f'\nMaximal {listType}: '+ max)
    print(f'\nMinimal {listType}: '+ min)
    print(f'\nAverage {listType}: '+ avg)
    print(f'\nStandard deviation ({listType}): '+'{0:.6f}'. format(stdev(list)))
    print('-------\n')

def CalculatePercentages(list):
    maxResult = '{0:.1f} %'. format(max(list)*100)
    minResult = '{0:.1f} %'. format(min(list)*100)
    avgResult = '{0:.1f} %'. format(mean(list)*100)
    return maxResult, minResult, avgResult

def CalculateDeimcals(list):
    maxResult = '{0:.3f}'. format(max(list))
    minResult = '{0:.3f}'. format(min(list))
    avgResult = '{0:.3f}'. format(mean(list))
    return maxResult, minResult, avgResult
