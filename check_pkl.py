import pickle

path1 = r'D:\ELECTRICAL_LOAD_FORECASTING\best_model_deployment\scaler_X.pkl'
with open(path1, 'rb') as f:
    obj1 = pickle.load(f)
print("scaler_X loaded OK:", type(obj1))

path2 = r'D:\ELECTRICAL_LOAD_FORECASTING\best_model_deployment\scaler_y.pkl'
with open(path2, 'rb') as f:
    obj2 = pickle.load(f)
print("scaler_y loaded OK:", type(obj2))

import sklearn
print("sklearn version:", sklearn.__version__)