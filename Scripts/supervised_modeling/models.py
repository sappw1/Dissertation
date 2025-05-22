import cuml
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def get_model(name, use_gpu, input_dim=None):
    try:
        if name == "logreg":
            if use_gpu:
                return cuml.LogisticRegression()
            else:
                return LogisticRegression(max_iter=1000)

        elif name == "svm":
            if use_gpu:
                return cuml.svm.SVC()
            else:
                return SVC(probability=True)

        elif name == "rf":
            if use_gpu:
                return cuml.ensemble.RandomForestClassifier()
            else:
                return RandomForestClassifier(n_estimators=100)

        elif name == "naive_bayes":
            if use_gpu:
                return cuml.naive_bayes.GaussianNB()
            else:
                return GaussianNB()

        elif name == "xgboost":
            params = {'tree_method': 'hist', "device": "cuda"} if use_gpu else {}
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)

        elif name == "neural_net":
            model = Sequential()
            model.add(Dense(128, input_dim=input_dim, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
            return model

    except Exception as e:
        print(f"[WARN] GPU fallback triggered for {name}: {e}")
        # Recursively retry with CPU
        return get_model(name, use_gpu=False, input_dim=input_dim)

