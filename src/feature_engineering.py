from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import numpy as np
import data_preprocessing as dp
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC

train = dp.load_train_data('../data/train.csv')
test = dp.load_test_data('../data/test.csv')

# Adding some basic new features

RMV = ['rainfall','id']
train['year_group'] = train['id']//365
train['temperature_range'] = train['maxtemp'] - train['mintemp']
train['seasonal_sin'] = np.sin(2 * np.pi * train['day'] / 365)    # Tracks seasonal behavior
test['year_group'] = test['id']//365
test['temperature_range'] = test['maxtemp'] - test['mintemp']
test['seasonal_sin'] = np.sin(2 * np.pi * test['day'] / 365)
FEATURES = [c for c in train.columns if not c in RMV]

if __name__ == "__main__":
    print("Our features are:")
    print( FEATURES )

# Smack together all features for future testing

INTERACT = []
for i,c1 in enumerate(FEATURES):
    for j,c2 in enumerate(FEATURES[i+1:]):
        n = f"{c1}_{c2}"
        train[n] = train[c1] * train[c2]
        test[n] = test[c1] * test[c2]
        INTERACT.append(n)

if __name__ == "__main__":
    print(f"There are {len(INTERACT)} interaction features:")
    print( INTERACT )

# Testing above features for best performance (change model as needed).

if __name__ == "__main__":
    ADD  = []
    best_auc = 0
    best_oof = None
    best_pred = None

        # FORWARD FEATURE SELECTION
        for k,col in enumerate(['baseline']+INTERACT):

            FOLDS = train.year_group.nunique()
            kf = GroupKFold(n_splits=FOLDS)

            oof_svc = np.zeros(len(train))
            pred_svc = np.zeros(len(test))

            if col!='baseline': ADD.append(col)

            # GROUP K FOLD USING YEAR AS GROUP
            for i, (train_index, test_index) in enumerate(kf.split(train, groups=train.year_group)):
            # TRAIN AND VALID DATA
                x_train = train.loc[train_index, FEATURES+ADD].copy()
                y_train = train.loc[train_index, "rainfall"]
                x_valid = train.loc[test_index, FEATURES+ADD].copy()
                y_valid = train.loc[test_index, "rainfall"]
                x_test = test[FEATURES+ADD].copy()

                # SVC WANTS STANDARIZED FEATURES
                for c in FEATURES + ADD:
                    m = x_train[c].mean()
                    s = x_train[c].std()
                    x_train[c] = (x_train[c] - m) / s
                    x_valid[c] = (x_valid[c] - m) / s
                    x_test[c] = (x_test[c] - m) / s
                    x_test[c] = x_test[c].fillna(0)

                # TRAIN SVC MODEL
                # LinearSVC does not support `predict_proba`, so we use decision_function to get scores
                model = LinearSVC(C=0.1)
                model.fit(x_train.values, y_train.values)

                # INFER OOF
                decision_values = model.decision_function(x_valid.values)
                oof_svc[test_index] = 1 / (1 + np.exp(-decision_values))  # Logistic transformation

                # INFER TEST
                decision_values_test = model.decision_function(x_test.values)
                pred_svc += 1 / (1 + np.exp(-decision_values_test))  # Logistic transformation

            # COMPUTE AVERAGE TEST PREDS
            pred_svc /= FOLDS

            # COMPUTE CV VALIDATION AUC SCORE
            true = train.rainfall.values
            m = roc_auc_score(true, oof_svc)

            if m > best_auc:
                print(f"NEW BEST with {col} at {m}")
                best_auc = m
                best_oof = oof_svc.copy()
                best_pred = pred_svc.copy()
            else:
                print(f"Worse with {col} at {m}")
                ADD.remove(col)

print("")
# Best AUC
if __name__ == "__main__":
    print(f"We achieved CV SVC AUC = {best_auc:.4f} adding {len(ADD)} interactions features:")
    print( ADD )

# Update dataframe with best features!

ADD = ['day_maxtemp', 'day_sunshine', 'day_winddirection', 'day_year_group', 'pressure_maxtemp', 'pressure_mintemp', 'pressure_temperature_range', 'maxtemp_temparature', 'maxtemp_dewpoint', 'maxtemp_year_group', 'maxtemp_seasonal_sin', 'temparature_dewpoint', 'temparature_cloud', 'temparature_winddirection', 'temparature_windspeed', 'temparature_year_group', 'mintemp_winddirection', 'dewpoint_winddirection', 'dewpoint_year_group', 'humidity_year_group', 'cloud_windspeed', 'windspeed_year_group']

difference = list(set(INTERACT) - set(ADD))

train = train.drop(columns=difference)
test = test.drop(columns=difference)
