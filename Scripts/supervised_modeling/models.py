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
    if name == "logreg":
        return cuml.LogisticRegression() if use_gpu else LogisticRegression(max_iter=1000)
    elif name == "svm":
        return cuml.SVC() if use_gpu else SVC(probability=True)
    elif name == "naive_bayes":
        return GaussianNB()
    elif name == "decision_tree":
        return cuml.DecisionTreeClassifier() if use_gpu else DecisionTreeClassifier()
    elif name == "rf":
        return cuml.RandomForestClassifier() if use_gpu else RandomForestClassifier(n_estimators=100)
    elif name == "xgboost":
        params = {'tree_method': 'gpu_hist'} if use_gpu else {}
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
    elif name == "neural_net":
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
        return model
