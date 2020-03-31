# Импорт бибилиотек

import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy import interp
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import chi2, mutual_info_classif, RFECV
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc, \
                            log_loss, roc_auc_score, average_precision_score, confusion_matrix

#----------------------запуск модели ---------------------------------#

import build_dataset
import params_setting
import prepare_dataset
import class_balance
import fit_predict


### создание датасета ###
CHURNED_START_DATE, CHURNED_END_DATE, INTER_LIST = params_setting.build_dataset_params()

build_dataset.build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                  churned_end_date=CHURNED_END_DATE,
                  inter_list=INTER_LIST,
                  raw_data_path='train/',
                  dataset_path='dataset/',
                  mode='train')

build_dataset.build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                  churned_end_date=CHURNED_END_DATE,
                  inter_list=INTER_LIST,
                  raw_data_path='test/',
                  dataset_path='dataset/',
                  mode='test')

train = pd.read_csv('dataset/dataset_raw_train.csv', sep=';')
test = pd.read_csv('dataset/dataset_raw_test.csv', sep=';')
print(train.shape, test.shape)

### обработка датасета ###
prepare_dataset.prepare_dataset(dataset=train, dataset_type='train')
prepare_dataset.prepare_dataset(dataset=test, dataset_type='test')

dataset = pd.read_csv('dataset/dataset_train.csv', sep=';')
X = dataset.drop(['user_id', 'is_churned'], axis=1)
y = dataset['is_churned']

X_mm = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=100)


### балансировка классов ###
X_train_balanced, y_train_balanced = class_balance.class_balance(X_train, y_train)

### обучение модели ###
fitted_clf_basic = fit_predict.xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)