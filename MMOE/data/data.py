import pandas as pd
from sklearn.model_selection import train_test_split

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
dfd = dfd.drop(dfd[dfd['native-country_ Holand-Netherlands']==1].index) # 18175 to keep the input features of train and test consistent
dfd.drop(columns='native-country_ Holand-Netherlands', inplace=True)

dfd.to_csv('marital-status-train.csv', index=False)

# cope with validation and test data
df = pd.read_csv('adult.test', names=cols)
df.drop([0], inplace=True)
df['age'] = df['age'].astype(int)
df = df.drop(df[df['workclass']==' ?'].index)
df = df.drop(df[df['occupation']==' ?'].index)
df = df.drop(df[df['native-country']==' ?'].index)
df = df.reset_index(drop=True)

df['income'] = df['income'].apply(lambda x: 1 if x == ' >50K.' else 0).astype(int)
df['marital-status'] = df['marital-status'].apply(lambda x: 1 if x == ' Never-married' else 0).astype(int) 

dfd = pd.get_dummies(df.drop(label_cols, axis=1)) # #features: 13 -> 97
dfd['marital-status'] = df['marital-status']
dfd['income'] = df['income']

dfv, dft = train_test_split(dfd, test_size=0.5, random_state=42)
dfv.to_csv('marital-status-valid.csv', index=False)
dft.to_csv('marital-status-test.csv', index=False)

# ddf = [x for x in dftcols if x not in dfdcols] # check if any diffences in columns of dfv, dft and dfd.