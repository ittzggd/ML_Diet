from klearn.model_selection import train_test_split


X = pd.read_csv("../../data/processed/personal_features_scaled.csv")
y = pd.read_csv("../../data/processed/personal_targets.csv")[["Recommended_Calories"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

