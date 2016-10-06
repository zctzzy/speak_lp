# -*- coding: utf-8 -*-
# This is a file created by Administrator at 2016/8/14
# Project Name: LinkPrediction
# Author:       chuanting zhang
# Email:        chuanting.zhang@gmail.com
# Redistribution of this code is permitted.

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.regularizers import l2, activity_l2
# from sklearn.neural_network import MLPClassifier

def create_baseline(bias1=0.01, bias2=0.001):
    model = Sequential()
    model.add(Dense(out_dim,
                    input_dim=in_dim,
                    init='normal',
                    activation='relu',
                    # bias=bias1
                    ))

    # model.add(Dense(in_dim,
    #                 init='normal',
    #                 activation='relu'))

    model.add(Dense(int(out_dim/2),
                    init='normal',
                    activation='relu',
                    # bias=bias2
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

def build_model_rbm():
    np.random.seed(12)
    rbm_estimators = list()
    # rbm = BernoulliRBM(random_state=12, verbose=0, n_components=in_dim)
    rbm = BernoulliRBM(random_state=np.random.randint(1, 100), verbose=0)
    lr = LogisticRegression()

    rbm.learning_rate = 0.0001
    # rbm.n_iter = 20
    # rbm.n_components = 50

    lr.C = 10.0

    rbm_estimators.append(('rbm', rbm))
    rbm_estimators.append(('lr', lr))

    return rbm_estimators

def build_model_rf():
    rf_estimator = list()
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=12)
    rf_estimator.append(('rf', rf))
    return rf_estimator

def build_model_svc():
    svc_estimator = list()

    svc = SVC(probability=True)
    svc_estimator.append(('svm', svc))
    return svc_estimator

def build_model_latent():
    latent_estimator = list()
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=20)
    latent_estimator.append(('rf', rf))
    return latent_estimator

def train_test(X, Y, choice):
    estimators = []
    if choice == 1:
        estimators = build_model_mlp()
    elif choice == 2:
        estimators = build_model_rbm()
    elif choice == 3:
        estimators = build_model_rf()
    elif choice == 4:
        estimators = build_model_svc()
    elif choice == 5:
        estimators = build_model_latent()

    clf = Pipeline(estimators)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0., 1., 30)

    auc_all = []

    num_of_exp = 20
    for i in range(1, num_of_exp+1):
        print "第%d次测试,%d/%d." % (i, i, num_of_exp)
        x_train, x_test, y_train, y_test = \
            train_test_split(X, Y,
                             test_size=0.2,
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

def unsupervised_method(attr, label):
    fpr_, tpr_, thresholds_ = metrics.roc_curve(label, attr)
    mean_tpr_ = 0.0
    mean_fpr_ = np.linspace(0., 1., 30)
    mean_tpr_ += interp(mean_fpr_, fpr_, tpr_)
    roc_auc_attr = metrics.auc(fpr_, tpr_)
    return mean_tpr_, roc_auc_attr


fp_hand = 'e:/speak_data/data2/balanced_sample/second_4hop_hand_1.csv'
fp_auto = 'e:/speak_data/data2/balanced_sample/second_4hop_auto_1.csv'
# fp_hand = 'e:/first_4hop_hand_new.csv'
# fp_auto = 'e:/first_4hop_auto_new.csv'
df_hand = pd.read_csv(fp_hand, header=0)
df_auto = pd.read_csv(fp_auto, header=0)

df_hand = df_hand.reindex(np.random.permutation(df_hand.index))
df_auto = df_auto.reindex(np.random.permutation(df_auto.index))

propflow = df_hand['propflow'].values
propflow = np.array(propflow).reshape((len(propflow), 1))

latent_auto = df_auto.values
latent_Y = latent_auto[:, -1].astype(int)
latent_X = latent_auto[:, 0:-1].astype(float)
latent_X = StandardScaler().fit_transform(latent_X)

data = df_hand.drop(['tag', 'aff', 'sp'], axis=1)
data = data.values
Y = data[:, -3].astype(int)
X = data[:, 0:-3].astype(float)
X = StandardScaler().fit_transform(X)

speak_data = df_hand.values
speak_Y = speak_data[:, -3].astype(int)
speak_X = speak_data[:, 0:-3].astype(float)
speak_X = StandardScaler().fit_transform(speak_X)

in_dim = speak_X.shape[1]
out_dim = 2*in_dim

results_fpr_tpr = []
final_fpr = np.linspace(0., 1., 30)
results_fpr_tpr.append(final_fpr)
# # MLP
# mlp_tpr, mlp_auc, mlp_std = train_test(speak_X, speak_Y, 1)
# print("MLP AUC: %.3f(%.3f)" % (mlp_auc, mlp_std))
# plt.plot(final_fpr, mlp_tpr, '--ro', label='MLP AUC: %.3f(%.3f)' % (mlp_auc, mlp_std))
# results_fpr_tpr.append(mlp_tpr)
#
# # RBM
# rbm_tpr, rbm_auc, rbm_std = train_test(X, Y, 2)
# print("RBM AUC: %.3f(%.3f)" % (rbm_auc, rbm_std))
# plt.plot(final_fpr, rbm_tpr, '--gs', label='RBM AUC: %.3f(%.3f)' % (rbm_auc, rbm_std))
# results_fpr_tpr.append(rbm_tpr)
#
# # RBM Full Feature
# rbm1_tpr, rbm1_auc, rbm1_std = train_test(speak_X, speak_Y, 2)
# print("RBM full AUC: %.3f(%.3f)" % (rbm1_auc, rbm1_std))
# plt.plot(final_fpr, rbm1_tpr, '--gs', label='RBM AUC(FULL): %.3f(%.3f)' % (rbm1_auc, rbm1_std))
# results_fpr_tpr.append(rbm1_tpr)
# RF
rf_tpr, rf_auc, rf_std = train_test(X, Y, 3)
print("RF AUC: %.3f(%.3f)" % (rf_auc, rf_std))
plt.plot(final_fpr, rf_tpr, '--c^', label='RF AUC: %.3f(%.3f)' % (rf_auc, rf_std))
results_fpr_tpr.append(rf_tpr)

# HPLP FULL
rf1_tpr, rf1_auc, rf1_std = train_test(speak_X, speak_Y, 3)
print("RF full AUC: %.3f(%.3f)" % (rf1_auc, rf1_std))
plt.plot(final_fpr, rf1_tpr, '--c^', label='RF AUC(FULL): %.3f(%.3f)' % (rf1_auc, rf1_std))
results_fpr_tpr.append(rf1_tpr)

# Propflow
propflow_tpr, propflow_auc = unsupervised_method(propflow, Y)
plt.plot(final_fpr, propflow_tpr, '--v', label='Propflow AUC: %.3f' %propflow_auc)
print('Propflow roc (area = %0.3f)' % propflow_auc)
results_fpr_tpr.append(propflow_tpr)

# node2vec+RF
# node2vec_tpr, node2vec_auc, node2vec_std = train_test(latent_X, latent_Y, 5)
# plt.plot(final_fpr, node2vec_tpr, '--s', label='node2vec AUC: %.3f(%.3f)' % (node2vec_auc, node2vec_std))
# print('node2vec AUC: %0.3f(%.3f))' % (node2vec_auc, node2vec_std))
# results_fpr_tpr.append(node2vec_tpr)

# result = pd.DataFrame(results_fpr_tpr)
# result.to_csv('e:/result_fpr.csv', index=False)
plt.legend()
plt.grid()
plt.show()
