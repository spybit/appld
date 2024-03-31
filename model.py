import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Importing data set
df = pd.read_csv("Liver_disease.csv")
df.drop_duplicates(keep="first", inplace=True)

# Filling NULL values
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

# handling categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.Gender = le.fit_transform(df.Gender)

df = df[['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']]

# spliting
x = df.drop(columns=['Dataset'], axis=1)
y = df['Dataset']

# solving imblancing
from imblearn.over_sampling import SMOTE
smk = SMOTE(random_state=42)
x_res, y_res = smk.fit_resample(x, y)
resampled_df = pd.DataFrame(x_res, columns=df.columns)

# data Scaling
from sklearn.preprocessing import MinMaxScaler
mn = MinMaxScaler()
x_data = mn.fit_transform(x_res)

# Train Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.25, random_state=42)

#LGBMClassifier
import lightgbm as lgb
model9 = lgb.LGBMClassifier()
model9.fit(x_train, y_train)

#a9 = model9.score(x_test, y_test)*100
#print("Accuracy is", f"{a9:.5f}")

# make pickle file
pickle.dump(model9, open("model.pkl", "wb"))