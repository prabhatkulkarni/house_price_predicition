# house_price_predicition
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
