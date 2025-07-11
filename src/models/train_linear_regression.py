import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


x =  pd.read_csv("../../data/processed/personal_features_scaled.csv")
y = pd.read_csv("../../data/processed/personal_features_targets.csv")[["Recommended_Calories"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=30)

reg = LinearRegression().fit(x_train, y_train)


y_test = reg.predict(x_test)