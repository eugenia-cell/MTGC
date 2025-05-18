
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def compute_metrics(y_pred,y_test):
    y_pred[y_pred<0]=0
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    return mae,np.sqrt(mse),r2

def regression(X_train,y_train,X_test,alpha):
    reg=linear_model.Ridge(alpha=alpha)
    X_train=np.array(X_train,dtype=float)
    y_train=np.array(y_train,dtype=float)
    reg.fit(X_train,y_train)

    y_pred=reg.predict(X_test)
    return y_pred

def kf_predict(X,Y):
    kf=KFold(n_splits=5)
    y_preds=[]
    y_truths=[]
    for train_index,test_index in kf.split(X):
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=Y[train_index],Y[test_index]
        y_pred=regression(X_train,y_train,X_test,1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds),np.concatenate(y_truths)

def predict_regression(embs,labels,display=False):
    y_pred,y_test=kf_predict(embs,labels)
    mae,rmse,r2=compute_metrics(y_pred,y_test)
    if display:
        print("MAE: ",mae)
        print("RMSE: ",rmse)
        print("R2: ",r2)
    return mae,rmse,r2

def lu_classify(emb,display=False):
    lu_label_filename="Data/eval_dataset/mh_cd.json"
    cd=json.load(open(lu_label_filename))
    cd_labels=np.zeros((180))
    for i in range(180):
        cd_labels[i]=cd[str(i)]
    
    n=12
    kmeans=KMeans(n_clusters=n,random_state=3,n_init=10)
    emb_labels=kmeans.fit_predict(emb)

    # normalized_mutual_info_score函数衡量的是两个数据标签集合之间的相似度。 NMI的值范围从0（无共享信息）到1（完美的相关性）。一个高的NMI值表示聚类结果与实际类别之间有较高的一致性。
    # ARI考虑了所有元素对（标签对）并计算在两个标签集合中这些对被分类为相同类或不同类的频率。ARI的值在-1（标签完全独立）和1（标签完全相同）之间。值为0表示聚类结果在随机情况下的平均水平。
    nmi=normalized_mutual_info_score(cd_labels,emb_labels)
    ars=adjusted_rand_score(cd_labels,emb_labels)
    if display:
        print("emb nmi: {:.3f}".format(nmi))
        print("emb ars: {:.3f}".format(ars))
    return nmi,ars

def do_tasks(embs,display=True):
    if display:
        print("Crime Count Prediction: ")
    crime_count_label=np.load("Data/eval_dataset/crime_counts_label.npy")
    crime_count_label=crime_count_label[:,0]
    crime_mae,crime_rmse,crime_r2=predict_regression(embs,crime_count_label,display=display)

    if display:
        print("Check-in Prediciton: ")
    check_in_label=np.load("Data/eval_dataset/check_in_label.npy")
    check_mae,check_rmse,check_r2=predict_regression(embs,check_in_label,display=display)

    if display:
        print("land Usage Prediction: ")
    nmi,ars=lu_classify(embs,display=display)
    
    return crime_mae, crime_rmse, crime_r2, check_mae, check_rmse, check_r2, nmi, ars
