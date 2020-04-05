### обучение модели ###

import xgboost as xgb

import evaluate_model

def xgb_fit_predict(X_train, y_train, X_test, y_test, X_predict):

    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight = 3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10)

    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    predict_proba_test = clf.predict_proba(X_test)
    predict_test = clf.predict(X_test)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluate_model.evaluation(y_test, predict_test, predict_proba_test[:, 1])
    predict_final = clf.predict(X_predict)
    return predict_final