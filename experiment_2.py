import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#from RBF_Svm_generalized_pinball import RBF_SVM_GP_sgd

from s_SVM_Hingeloss import SVM_Hinge_sgd_2
from s_SVM_insensitive import SVM_Insensitive_BFGS_2
from s_SVM_generalized_pinball import SVM_GP_sgd_2

from RBF_s_SVM_Hingeloss import RBF_SVM_Hinge_sgd_2
from RBF_s_SVM_generalized_pinball import RBF_SVM_GP_sgd_2
from RBF_s_SVM_insensitive import RBF_SVM_Insensitive_BFGS_2

#from SVM_Hingeloss_2 import SVM_Hinge_sgd_2

data = np.genfromtxt(r'C:\Users\Kook\Desktop\SVM_Uncertain_2\data\spectfheart.dat',
                      skip_header=49,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

#data = data[:,:-1]

x = data[:,:-1]
t = data[:,-1]
t[t == 0] = -1


from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)


import noiseADD as na
#x = na.withnoise(x, r=0.05)
#x = na.withnoise(x, r=0.1)
#x = na.withnoise(x, r=0.15)
#x = na.withnoise(x, r=0.20)
#x  = na.withnoise(x, r=0.30)
#x = na.withnoise(x, r=0.50)
#x = na.withnoise(x, r=0.70)

#x = np.array(x)
#t = np.array(t)

x_train, x_test, y_train, y_test = train_test_split(x,t, test_size=0.3, random_state=2, stratify=t)


#model = SVM_Hinge_sgd(max_epochs = 50, n_batches = 64, C=0.0001)
#model = SVM_Hinge_sgd_3(max_epochs = 50, n_batches = 64, C=0.03125)
#model = SVM_Insensitive_BFGS_2(max_epochs = 50, n_batches = 64, C=0.03125, tau =0.8, epsilon = 1)
#model = SVM_GP_sgd_2(C = 0.03125, max_epochs = 50 , n_batches = 64, tau_1 = 0.8, tau_2 = 0.4, epsilon_1 = 0.8, epsilon_2 = 0.8)

#model = RBF_SVM_Hinge_sgd_2(max_epochs = 50, n_batches = 64, C= 0.0625)
#model = RBF_SVM_Insensitive_BFGS_2(max_epochs = 50, n_batches = 64, C=0.125, tau =0.2  , epsilon = 0.4)
model = RBF_SVM_GP_sgd_2(C = 1, max_epochs = 50 , n_batches = 64, tau_1 = 1, tau_2 = 0.8, epsilon_1 = 1, epsilon_2 = 0.8)

model.fit(x_train, y_train)

x_test_new = model.construct_x_uncertain(x_test, y_test)
y_predict = model.predict(x_test_new)

acc = accuracy_score(y_test, y_predict)
print(classification_report(y_test, y_predict))
print(acc)
#print(classification_report)