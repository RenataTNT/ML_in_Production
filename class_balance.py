### балансировка классов ###

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def class_balance(X, y):
    sm = SMOTE(random_state=42, sampling_strategy=0.3, n_jobs=1)
    X_train_balanced, y_train_balanced = sm.fit_sample(X, y)
    return X_train_balanced, y_train_balanced