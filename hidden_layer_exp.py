# -*- coding: utf-8 -*-
# This is a file created by Administrator at 2016/8/27
# Project Name: LinkPrediction
# Author:       chuanting zhang
# Email:        chuanting.zhang@gmail.com
# Redistribution of this code is permitted.

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp


def create_baseline():
    model = Sequential()
    model.add(Dense(out_dim,
                    input_dim=in_dim,
                    init='normal',
                    activation='relu'
                    ))

    model.add(Dense(out_dim/2,
                    init='normal',
                    activation='relu'
                    ))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def build_model_mlp():
    np.random.seed(11)
    mlp_estimators = list()
    mlp_estimators.append(('scaler', StandardScaler()))
    mlp_estimators.append(('mlp',
                       KerasClassifier(build_fn=create_baseline,
                                       nb_epoch=100,
                                       batch_size=50,
                                       verbose=0)))
    return mlp_estimators

def train_test(X, Y, ratio):
    estimators = build_model_mlp()
    clf = Pipeline(estimators)
    # clf = RandomForestClassifier(n_jobs=-1, n_estimators=12)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0., 1., 30)

    auc_all = []

    num_of_exp = 20
    for i in range(1, num_of_exp+1):
        print "第%d次测试,%d/%d." % (i, i, num_of_exp)
        x_train, x_test, y_train, y_test = \
            train_test_split(X, Y,
                             test_size=ratio,
                             random_state=np.random.randint(1, 100))
        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        auc_all.append(metrics.roc_auc_score(y_test, y_pred))

    mean_tpr /= num_of_exp

    auc_array = np.array(auc_all)
    auc = auc_array.mean()
    auc_std = auc_array.std()
    mean_tpr[-1] = 1.0
    return mean_tpr, auc, auc_std


fp_hand = 'e:/speak_data/data2/balanced_sample/first_4hop_hand_1.csv'

df_hand = pd.read_csv(fp_hand, header=0)

df_hand = df_hand.reindex(np.random.permutation(df_hand.index))

df_hand = df_hand.drop(['sp'], axis=1)
data = df_hand.values
Y = data[:, -3].astype(int)
X = data[:, 0:-3].astype(float)
X = StandardScaler().fit_transform(X)

in_dim = X.shape[1]
out_dim = 2*in_dim

final_fpr = np.linspace(0., 1., 30)

# MLP
test_size = 0.2
mlp_tpr, mlp_auc, mlp_std = train_test(X, Y, test_size)
print("MLP AUC: %.3f(%.3f)" % (mlp_auc, mlp_std))
plt.plot(final_fpr, mlp_tpr, '--ro', label='MLP AUC: %.3f(%.3f)' % (mlp_auc, mlp_std))

plt.legend()
plt.grid()
plt.show()




