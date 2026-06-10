# %% [markdown]
# # House Price Prediction using Linear Regression
# 
# ## About this Project
# This project predicts house prices using Linear Regression on the King County, 
# Washington (USA) house sales dataset. I built and compared separate single-feature 
# models (bedrooms, bathrooms, sqft_living) and one multi-feature model to understand 
# how each factor influences price.
# 
# ## Dataset
# - **Source:** King County House Sales dataset (available on Kaggle)
# - **Rows:** ~4,600 house sale records
# - **Key features used:** bedrooms, bathrooms, sqft_living
# - **Target variable:** price (in USD)
# 
# ## What I did
# 1. Loaded and explored the dataset using Pandas
# 2. Trained 3 separate Linear Regression models (one feature each)
# 3. Trained 1 multi-feature model combining all three
# 4. Compared R² scores to see which feature predicts price best
# 5. Visualized actual vs predicted prices using Matplotlib

# %%
import pandas as pd
import numpy as  np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
house_df = pd.read_csv("data.csv")

# %%
feature_1 = house_df["bedrooms"].values
feature_2 = house_df["bathrooms"].values
feature_3 = house_df["sqft_living"].values
new_feature_1 = feature_1.reshape(-1,1)
new_feature_2 = feature_2.reshape(-1,1)
new_feature_3 = feature_3.reshape(-1,1)
label = house_df["price"].values

# %%
feature = house_df[["bedrooms","bathrooms","sqft_living"]]
X_train, X_test, y_train, y_test = train_test_split(feature,label,test_size=0.3,random_state=42)
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)
y_predict = l_reg.predict(X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
X1_train, X1_test, y1_train, y1_test = train_test_split(new_feature_1,label,test_size=0.3,random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(X1_train,y1_train)
y1_predict = linear_reg.predict(X1_test)

# %%
X2_train, X2_test, y2_train, y2_test = train_test_split(new_feature_2,label,test_size=0.3,random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(X2_train,y2_train)
y2_predict = linear_reg.predict(X2_test)

# %%
X3_train, X3_test, y3_train, y3_test = train_test_split(new_feature_3,label,test_size=0.2,random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(X3_train,y3_train)
y3_predict = linear_reg.predict(X3_test)

# %%
plt.figure(figsize=(10,8))  
plt.subplot(1,3,1)
plt.scatter(X1_test, y1_test , color="blue")
plt.plot(X1_test, y1_predict,color ="black")
plt.ylim(0, 0.5e7)
plt.xlabel("BEDROOMS")
plt.ylabel("PRICE(in crores)")
plt.subplot(1,3,2)
plt.scatter(X2_test, y2_test , color="red")
plt.plot(X2_test, y2_predict,color ="black")
plt.ylim(0, 0.5e7)
plt.xlabel("BATHROOMS")
plt.ylabel("PRICE(in crores)")
plt.subplot(1,3,3)
plt.scatter(X3_test, y3_test , color="yellow")
plt.plot(X3_test, y3_predict,color ="black")
plt.ylim(0, 0.5e7)
plt.xlabel("SQUARE FOOT")
plt.ylabel("PRICE(in crores)")
plt.xlim(0,6000)
plt.show()

# %%
plt.figure(figsize=(8,8))

plt.scatter(y_test, y_predict)

# Perfect prediction line
min_val = min(min(y_test), min(y_predict))
max_val = max(max(y_test), max(y_predict))

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color = "black"
)



plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.xlim(0,0.15e7)
plt.ylim(0.5e5,1.5e6)

plt.show()

# %%
print("Bedrooms R²:", linear_reg.score(X1_test, y1_test))
print("Bathrooms R²:", linear_reg.score(X2_test, y2_test))
print("Sqft R²:", linear_reg.score(X3_test, y3_test))
print("Multi-feature R²:", l_reg.score(X_test, y_test))



