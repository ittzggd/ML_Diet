import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
# import sklearn.preprocessing import MinMaxScaler


file_path = "../../data/raw/Personalized_Diet_Recommendations.csv"

df = pd.read_csv(file_path)
df.drop(columns = ["Patient_ID"], inplace=True)

df = df.dropna()

categorical_cols = [
    "Gender", "Chronic_Disease", "Smoking_Habit", "Alcohol_Consumption", "Dietary_Habits", "Preferred_Cuisine", "Food_Aversions", "Genetic_Risk_Factor"
]

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


input_features = [
    "Age", "Gender", "Height_cm", "Weight_kg", "BMI",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
    "Cholesterol_Level", "Blood_Sugar_Level", "Daily_Steps",
    "Exercise_Frequency", "Sleep_Hours", "Smoking_Habit",
    "Alcohol_Consumption", "Dietary_Habits", "Preferred_Cuisine",
    "Chronic_Disease", "Genetic_Risk_Factor", "Food_Aversions"
]

target_features = [
    "Recommended_Calories", "Recommended_Protein", 
    "Recommended_Carbs", "Recommended_Fats"
]

x = df[input_features]
y = df[target_features]

scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)


x_scaled_df.to_csv("../../data/processed/personal_features_scaled.csv", index=False)
y.to_csv("../../data/processed/personal_features_targets.csv", index=False)

