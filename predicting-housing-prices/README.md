# Supplemental activity for day 3 (08.07.2021) - Predicting housing prices in Iowa

## Description
This model uses DecisionTreeRegressor from sklearn to predict the housing prices of Iowa based on the the following inputs:
- Lot Area
- Build Year
- Area of the 1st floor
- Area of the 2nd floor
- Number of bathrooms
- Number of bedrooms above ground
- Number of rooms above ground

## Reflection
Since the model is making predictions it has already seen, it's prediction accuracy will be much greater than it's actual accuracy. The model has "memorized" the data and therefore, is not using learned relationships between the inputs and outputs. 