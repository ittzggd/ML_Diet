import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


file_path = "../../data/raw/nutrition_dataset1 - food.csv"
df = pd.read_csv(file_path)

selected_columns = {
    "Description" : "name",
    "Data.Kilocalories" : "kcal",
    "Data.Protein" : "Protein",
    "Data.Fat.Total Lipid": "fat",
    "Data.Carbohydrate" : "carbs",
    "Data.Fiber": "fiber",
    "Data.Sugar Total" : "sugar",
    "Data.Cholesterol" : "cholesterol",
    "Data.Choline" : "choline",
    "Data.Major Minerals.Calcium" : "calcium",
    "Data.Major Minerals.Sodium" :  "sodium",
    "Data.Maㅊjor Minerals.Iron" : "iron",
    "Data.Major Minerals.Magnesium": "magnesium",
    "Data.Major Minerals.Phosphorus" : "phosphorus",
    # "Data.Vitamins "
}

food_df = df[list(selected_columns.keys())].rename(columns=selected_columns)

food_df = food_df.dropna()

food_df = food_df.drop_duplicates(subset=["name"])

food_df.reset_index(drop=True, inplace=True)

scaler = MinMaxScaler()
numeric_columns = food_df.columns.drop("name")
food_df[numeric_columns] = scaler.fit_transform(food_df[numeric_columns])


food_df.to_csv("../../data/processed/food_data_processed.csv", index=False)
# print(food_df.head())