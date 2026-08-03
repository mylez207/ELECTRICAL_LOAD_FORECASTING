path = r'D:\ELECTRICAL_LOAD_FORECASTING\app.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace pickle import with joblib import
content = content.replace('import pickle\n', 'import joblib\n')

# Replace the three specific pickle.load patterns with joblib.load equivalents
content = content.replace(
    "with open(path, 'rb') as f:\n                    assets['model'] = pickle.load(f)",
    "assets['model'] = joblib.load(path)"
)
content = content.replace(
    "with open(path, 'rb') as f:\n                assets[key] = pickle.load(f)",
    "assets[key] = joblib.load(path)"
)
content = content.replace(
    "with open(feat_path, 'rb') as f:\n            assets['feature_cols'] = pickle.load(f)",
    "assets['feature_cols'] = joblib.load(feat_path)"
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    new_content = f.read()

print("Remaining 'pickle.load' occurrences:", new_content.count('pickle.load'))
print("New 'joblib.load' occurrences:", new_content.count('joblib.load'))