# Day 2 Lab: Answers
## 1. Baseline. How much does your regression beat predict the mean (RMSE and R2
)? Is it

worth deploying?
## 2. Accuracy paradox
What accuracy and recall does the dummy classier get on big\_quake?
Why is accuracy the wrong headline metric?

## 3. Threshold
What threshold did your search choose, compared with 0.5? Which error would
you rather make in an earthquake alert system, and why?

##4. Leakage check
Remove one leaky column name from config.LEAKY, rerun the ETL and
train, and compare scores. What happened, and why is the better number a lie?

## 5. Complexity
Train with model poly at degree 1, 2 and 4. When does performance on the
test set fall below the training result?

## 6. Gradient descent
Do your GDLinearRegression coecients match LinearRegression on
the notebook data? What changes if you skip scaling?
