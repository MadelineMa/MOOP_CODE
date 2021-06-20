import pandas as pd

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('adult.data', names=cols) # 32,561 entries with unknowns

# unknowns ars marked with '?'
df = df.drop(df[df['workclass']==' ?'].index)
df = df.drop(df[df['occupation']==' ?'].index)
df = df.drop(df[df['native-country']==' ?'].index)
df = df.reset_index(drop=True)

label_cols = ['income', 'marital-status']

df['income'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0).astype(int)
df['marital-status'] = df['marital-status'].apply(lambda x: 1 if x == ' Never-married' else 0).astype(int) 

dfd = pd.get_dummies(df.drop(label_cols, axis=1)) # #features: 13 -> 97
dfd['marital-status'] = df['marital-status']
dfd['income'] = df['income']

dfd.to_csv('marital-status.csv')
