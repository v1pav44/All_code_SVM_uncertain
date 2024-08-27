import numpy as np
import pandas as pd
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn import model_selection 
from sklearn.model_selection import train_test_split

from SVM_generalized_pinball import SVM_GP_sgd
from SVM_Hingeloss import SVM_Hinge_sgd
from SVM_insensitive import SVM_Insensitive_BFGS

from s_SVM_Hingeloss import SVM_Hinge_sgd_2
from s_SVM_insensitive import SVM_Insensitive_BFGS_2
from s_SVM_generalized_pinball import SVM_GP_sgd_2

from t_SVM_Hingeloss import SVM_Hinge_sgd_3

from RBF_s_SVM_Hingeloss import RBF_SVM_Hinge_sgd_2
from RBF_s_SVM_generalized_pinball import RBF_SVM_GP_sgd_2
from RBF_s_SVM_insensitive import RBF_SVM_Insensitive_BFGS_2

data = np.genfromtxt(r'C:\Users\Kook\Desktop\SVM_Uncertain_2\data\ionosphere.dat',
                      skip_header=38,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

#data = data[:,:-1]

x = data[:,:-1]
t = data[:,-1]
#t[t == 0] = -1


from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

import noiseADD as na
x = na.withnoise(x, r=0.05)
#x = na.withnoise(x, r=0.1)
#x = na.withnoise(x, r=0.15)
#x = na.withnoise(x, r=0.20)
#x  = na.withnoise(x, r=0.30)
#x = na.withnoise(x, r=0.50)
#x = na.withnoise(x, r=0.70)

#x = np.array(x)
#t = np.array(t)

#constructs an appropximate mapping dataset
# rbf_feature = RBFSampler(gamma=100, random_state=1)
# x = rbf_feature.fit_transform(x)

x_train, x_test, t_train, t_test = train_test_split(x,t, test_size=0.3, random_state=2, stratify=t)

'''
                            10-fold cross validation
'''
# from sklearn.model_selection import KFold


# n_folds = 10
# cv = KFold(n_splits=n_folds, shuffle=True, random_state=60031073)

# acc_train = np.empty((n_folds))
# acc_test = np.empty((n_folds))
# processingtime = np.empty((n_folds))
# conf_M_train = np.empty((n_folds,2,2))
# conf_M_test = np.empty((n_folds,2,2))
# #result = np.empty((n_folds))
# #result = []
print('#'*60)


models = [
    #('SVM_Hinge_sgd_3', SVM_Hinge_sgd_3(max_epochs = 10, n_batches = 64, C=0.0625)),
    #('SVM_Insensitive_BFGS_2', SVM_Insensitive_BFGS_2(max_epochs = 50, n_batches = 64, C=0.25, tau =0.8, epsilon = 1)),
    #('SVM_GP_sgd_2', SVM_GP_sgd_2(C = 0.0625, max_epochs = 3 , n_batches = 64, tau_1 = 0.8, tau_2 = 0.4, epsilon_1 = 0.8, epsilon_2 = 0.8)),
    ('RBF_SVM_Hinge_sgd_2', RBF_SVM_Hinge_sgd_2(max_epochs = 10, n_batches = 64, C= 0.0625)),
    #('RBF_SVM_Insensitive_BFGS_2', RBF_SVM_Insensitive_BFGS_2(max_epochs = 50, n_batches = 64, C=0.125, tau =0.2  , epsilon = 0.4)),
    # ('RBF_SVM_GP_sgd_2', RBF_SVM_GP_sgd_2(C = 0.03125, max_epochs = 50 , n_batches = 64, tau_1 = 0.8, tau_2 = 0.4, epsilon_1 = 0.8, epsilon_2 = 0.8))
 
]

dfs = []
results = []
names = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['positive', 'negative']
for name, model in  models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(model, x_train, t_train, cv=kfold, scoring=scoring)
    clf = model.fit(x_train, t_train)
    y_pred = clf.predict(x_test)
    print(name)
    #print(classification_report(t_[test], y_pred, target_names=target_names))
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    print(np.array(cv_results).mean())
    this_df['model'] = name
    dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
print(final)

# dfs = []
# results = []
# names = []
# scoring = ['accuracy']
# #scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
# target_names = ['positive', 'negative']
# for name, model in models:
#     kfold = model_selection.KFold(n_splits = 10, shuffle=True, random_state=191038)
#     cv_results = model_selection.cross_validate(model, x, t, cv=kfold, scoring=scoring)
#     #clf = model.fit(M, p)
#     #y_pred = clf.predict(test_data)
#     print(name)
#     #print(classification_report(p_test, y_pred, target_names=target_names))
#     #test = accuracy_score(p_test, y_pred)
#     results.append(cv_results)
#     names.append(name)
#     this_df = pd.DataFrame(cv_results)
#     this_df['model'] = name
#     #acc = this_df['test_accuracy']
#     #print(acc)
#     #print(np.mean(acc))
#     #print('%0.4f a accuracy with a standard diviation of %0.4f' % (cv_results.mean(), cv_results.std()))
#     Avg_accuracy = np.mean(cv_results)
#     #Std = cv_results.std()
#     #this_df['S.D.'] = Std
#     this_df['Average accuracy'] = Avg_accuracy
#     #this_df['Test'] = test
#     dfs.append(this_df)
#     final = pd.concat(dfs, ignore_index=True)
    

# final.to_csv('final.csv') 
# print('Dataframe is written to csv file Successfully.')
# print(dfs)
# # #print(results)
# #print(final)












      
'''
                            Import evaluation package
                            
# '''
# from sklearn.model_selection import cross_val_score
# kfold = KFold(n_splits = 10, shuffle=True, random_state=6003073)

# #model = RBF_SVM_Hinge_sgd_2(max_epochs = 10, n_batches = 64, C= 0.0625)
# #model = RBF_SVM_Insensitive_BFGS_2(max_epochs = 20, n_batches = 64, C=0.125, tau =2  , epsilon = 0.4)
# model = RBF_SVM_GP_sgd_2(C = 0.0625, max_epochs = 50 , n_batches = 64, tau_1 = 0.8, tau_2 = 0.4, epsilon_1 = 0.8, epsilon_2 = 0.8)


# cv_results = cross_val_score(model, x, t, cv=kfold, scoring='accuracy')
# print(cv_results)
# print('%0.4f a accuracy with a standard diviation of %0.4f' % (cv_results.mean(), cv_results.std()))
#acc = np.mean(cv_results)
#print('acc')

# # import time
# # from sklearn.metrics import accuracy_score, confusion_matrix
# # #############################################
# # #start Cross Validation
# # for i, (train, test) in enumerate(cv.split(x)):
# #     #choose model
    
# #     ##############################     Linear Case
    
# #     #model = SVM_Hinge_sgd(max_epochs = 10, n_batches = 64, C=100)
# #     #model = SVM_Insensitive_BFGS(max_epochs = 3, n_batches = 64, C=0.5, tau =0.2, epsilon = 0.4)
# #     #model = SVM_GP_sgd(C = 4, max_epochs = 50, n_batches = 64, tau_1 = 0.8, tau_2 = 0.2, epsilon_1 = 0.4, epsilon_2 = 0.2)
    
# #     #model = SVM_Hinge_sgd_2(max_epochs = 50, n_batches = 64, C=10)
# #     #model = SVM_Hinge_sgd_3(max_epochs = 50, n_batches = 64, C=0.0625)
# #     #model = SVM_Insensitive_BFGS_2(max_epochs = 50, n_batches = 64, C=0.25, tau =0.8, epsilon = 1)
# #     #model = SVM_GP_sgd_2(C = 0.25, max_epochs = 50, n_batches = 64, tau_1 = 0.8, tau_2 = 0.2, epsilon_1 = 0.8, epsilon_2 = 0.8)
    
# #     #############################     Nonlinear Case   
    
# #     #C in linear case and lamba are the same but using different alphabet in Hingeloss_sgd
# #     model = RBF_SVM_Hinge_sgd_2(max_epochs = 10, n_batches = 64, C= 1)
# #     #model = RBF_SVM_Insensitive_BFGS_2(max_epochs = 20, n_batches = 64, C=0.125, tau =2  , epsilon = 0.4)
# #     #model = RBF_SVM_GP_sgd_2(C = 2, max_epochs = 20 , n_batches = 64, tau_1 = 0.25, tau_2 = 0.2, epsilon_1 = 0.25, epsilon_2 = 0.25)
    
# #     start = time.time()
    
# #     #cv_results = model_selection.cross_validate(model, x[train], t[train], cv=10, scoring=scoring)
# #     #result.append(np.mean(cv_results, axis=0))

# #     #modeling
# #     #model.fit(x[train], t[train])
# #     model.fit(x[train], t[train]) 

# #     #cpu time used
# #     processingtime[i] = time.time() - start

# #     # x[train] = model.construct_x_uncertain(x[train], t[train])
# #     # x[test] = model.construct_x_uncertain(x[test], t[test])


# #     #prediction
# #     y_train = model.predict(x[train])
# #     y_test = model.predict(x[test])
    
# #     #evaluation model
# #     acc_train[i]= accuracy_score(t[train], y_train)
# #     acc_test[i] = accuracy_score(t[test], y_test)

# #     # confusion matrix
# #     conf_M_train[i] = confusion_matrix(t[train], y_train)
# #     conf_M_test[i] = confusion_matrix(t[test], y_test)
# #     #w_b = model.w_b
# #     #df = pd.DataFrame(w_b)

# # #display result
# # #df.to_csv(os.path.join(r'w_b.csv'))
# # import collections
# # #print(x[train].shape)
# # #print(t[train].shape)
# # print('CPU time = %.4f and its S.D. = %.4f' % (np.mean(processingtime), np.std(processingtime)))
# # print('TRUTH : ',collections.Counter(t[train]))
# # print('PREDICT : ',collections.Counter(y_train))
# # print("Training accuracy = %.4f and its S.D. = %.4f" % (np.mean(acc_train), np.std(acc_train)))
# # print('Training confusion matirx: \n', np.mean(conf_M_train, axis=0))
# # print('TRUTH : ',collections.Counter(t[test]))
# # print('PREDICT : ',collections.Counter(y_test))
# # print("Test accuracy     = %.4f and its S.D. = %.4f" % (np.mean(acc_test), np.std(acc_test)))
# # print('Test confusion matirx: \n', np.mean(conf_M_test, axis=0))
# # #result = np.vstack(result)
# # #print('cv_result: \n', np.mean(result, axis=0))
# # print('#'*60)