# Импорт бибилиотек

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split


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
prepare_dataset.prepare_dataset(INTER_LIST=INTER_LIST, dataset=train, dataset_type='train')
prepare_dataset.prepare_dataset(INTER_LIST=INTER_LIST, dataset=test, dataset_type='test')

# обучающий датасет #
dataset = pd.read_csv('dataset/dataset_train.csv', sep=';')
X = dataset.drop(['user_id', 'is_churned'], axis=1)
y = dataset['is_churned']

# датасет для финального предсказания #
dataset_predict=pd.read_csv('dataset/dataset_test.csv', sep=';')
X_predict=dataset_predict.drop(['user_id'], axis=1)

print('X', X.shape)
print('X_predict', X_predict.shape)

# масштабирование признаков #
X_mm = MinMaxScaler().fit_transform(X)
X_mm_predict = MinMaxScaler().fit_transform(X_predict)

X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=100)


### балансировка классов ###
X_train_balanced, y_train_balanced = class_balance.class_balance(X_train, y_train)

### обучение модели и предсказания ###
y_predict = fit_predict.xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test, X_mm_predict)

#предсказания в csv-файл#

final_prediction=pd.DataFrame(columns=['user_id','is_churned'])
final_prediction['user_id']=dataset_predict['user_id']
final_prediction['is_churned']=y_predict
final_prediction.to_csv('dataset/final_prediction.csv', sep=';', index=False)