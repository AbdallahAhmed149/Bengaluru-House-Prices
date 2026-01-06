import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

bengaluru_house_data = pd.read_csv("Kaggle Bengaluru House Price\\bengaluru_house_prices.csv")

print(bengaluru_house_data.shape)
print(bengaluru_house_data.info())
print(bengaluru_house_data.isna().sum())
print(bengaluru_house_data.head())
print(bengaluru_house_data.nunique())

# -------------------------------------------------------------------------------------------

def convert_sqft_to_float(sqft):
    try:
        if isinstance(sqft, (int, float)):
            return float(sqft)

        sqft = str(sqft).strip()
        if "-" in sqft:
            low, high = map(float, sqft.split("-"))
            return (low + high) / 2
        return float(sqft)

    except (ValueError, TypeError):
        return np.nan

bengaluru_house_data["area_type"] = bengaluru_house_data["area_type"].str.strip("  Area")
bengaluru_house_data["total_sqft"] = bengaluru_house_data["total_sqft"].apply(convert_sqft_to_float)
bengaluru_house_data["size"] = bengaluru_house_data["size"].str.extract(r"(\d+)")
bengaluru_house_data["size"] = bengaluru_house_data["size"].astype(float)

impute = SimpleImputer(strategy="most_frequent")
object_cols = bengaluru_house_data.select_dtypes("object").columns
bengaluru_house_data[object_cols] = impute.fit_transform(bengaluru_house_data[object_cols])

impute = SimpleImputer(strategy="median")
numeric_columns = bengaluru_house_data.select_dtypes("number").columns
bengaluru_house_data[numeric_columns] = impute.fit_transform(bengaluru_house_data[numeric_columns])

sns.heatmap(data=bengaluru_house_data.corr(numeric_only=True), annot=True)
plt.show()

fig, ax = plt.subplots(len(numeric_columns), 1, figsize=(7, 18), dpi=95)
for i, col in enumerate(numeric_columns):
    ax[i].boxplot(bengaluru_house_data[col], vert=False)
    ax[i].set_ylabel(col)
plt.show()

for column in numeric_columns:
    sns.kdeplot(data=bengaluru_house_data, x=column)
    plt.show()

for column in numeric_columns:
    lower_bound = bengaluru_house_data[column].quantile(0.01)
    upper_bound = bengaluru_house_data[column].quantile(0.99)
    bengaluru_house_data[column] = bengaluru_house_data[column].clip(
        lower=lower_bound, upper=upper_bound
    )

for column in numeric_columns:
    sns.kdeplot(data=bengaluru_house_data, x=column)
    plt.show()

sns.boxplot(data=bengaluru_house_data)
plt.show()

location_avg = bengaluru_house_data.groupby("location", as_index=False)["price"].mean()
location_avg.sort_values("price", inplace=True, ascending=False)
print(location_avg)

# -------------------------------------------------------------------------------------------

ordinal_encoding = OrdinalEncoder()
bengaluru_house_data["area_type_encoding"] = ordinal_encoding.fit_transform(bengaluru_house_data[["area_type"]])

target_encoding = TargetEncoder()
bengaluru_house_data["availability_encoding"] = target_encoding.fit_transform(bengaluru_house_data["availability"], bengaluru_house_data["price"])

sns.violinplot(data=bengaluru_house_data, x="availability", y="price")
plt.xticks(rotation=90)
plt.show()

# -------------------------------------------------------------------------------------------

X = bengaluru_house_data.drop("price", axis=1)
y = bengaluru_house_data["price"]

lasso_cv = LassoCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
lasso_cv.fit(X, y)
print(f"Best alpha: {lasso_cv.alpha_}")

selected_features = X.columns[lasso_cv.coef_ != 0].tolist()

X = bengaluru_house_data[selected_features]
y = bengaluru_house_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_validation = cross_val_score(model, X_train, y_train, cv=kf)
print(np.mean(cross_validation))

y_pred = model.predict(X_test)
print(model.score(X_train, y_train))
print(r2_score(y_test, y_pred))
