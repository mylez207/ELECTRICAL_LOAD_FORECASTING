path = r'D:\ELECTRICAL_LOAD_FORECASTING\app.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

before = content

content = content.replace(
    "'District Load (Short-Term Lags)</div>'",
    "'District Load</div>'"
)
content = content.replace(
    "'Statistics (Short-Term)</div>'",
    "'Statistics</div>'"
)
content = content.replace(
    "'Memory (NEW)</div>'",
    "'Memory</div>'"
)
content = content.replace(
    "'Socio-Economic Drivers (NEW)</div>'",
    "'Socio-Economic Drivers</div>'"
)

changed = (content != before)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("File changed:", changed)
print("Remaining 'Short-Term Lags':", content.count('Short-Term Lags'))
print("Remaining '(Short-Term)':", content.count('(Short-Term)'))
print("Remaining 'Memory (NEW)':", content.count('Memory (NEW)'))
print("Remaining 'Drivers (NEW)':", content.count('Drivers (NEW)'))