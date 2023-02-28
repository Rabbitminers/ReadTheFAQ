import joblib
import pandas as pd

clf = joblib.load('data/model.joblib')
vec = joblib.load('data/vec.joblib')

X_new = vec.transform(['Text1', 'I cant wait for extended bogeys'])

y_pred = clf.predict(X_new)

print(y_pred)