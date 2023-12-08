import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

file = pd.read_csv("/Users/shreyamhemanta/Documents/GitHub/Credit-Card-Fraud-Detection-2023-model/creditcard_2023.csv")

print(file)

x = file.iloc[:,:-1]
y = file.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33, random_state=42)

ss = StandardScaler()
x_scaled = ss.fit_transform(x)
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(x_scaled, y, train_size=0.33, random_state=42)

mm = MinMaxScaler()
x_min_max_scaled = mm.fit_transform(x)
x_train_min_max_scaled, x_test_min_max_scaled, y_train_min_max_scaled, y_test_min_max_scaled = train_test_split(x_min_max_scaled, y, train_size=0.33, random_state=42)

print("\nUnscaled Linear Regression")
lin = LinearRegression()
lin.fit(x_train, y_train)
y_pred_lin = lin.predict(x_test)
acc_lin = (1 - mean_absolute_error(y_test, y_pred_lin))*100
print(y_pred_lin, "\n", acc_lin)

print("\nStandard Scaled Linear Regression")
lin_scaled = LinearRegression()
lin_scaled.fit(x_train_scaled, y_train_scaled)
y_pred_scaled_lin = lin_scaled.predict(x_test_scaled)
acc_scaled_lin = (1 - mean_absolute_error(y_test_scaled, y_pred_scaled_lin))*100
print(y_pred_scaled_lin, "\n", acc_scaled_lin)

print("\nMin-Max Scaled Linear Regression")
lin_min_max_scaled = LinearRegression()
lin_min_max_scaled.fit(x_train_min_max_scaled,y_train_min_max_scaled)
y_pred_min_max_scaled_lin = lin_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_lin = (1 - mean_absolute_error(y_test_min_max_scaled, y_pred_min_max_scaled_lin))*100
print(y_pred_min_max_scaled_lin, "\n", acc_min_max_scaled_lin)

print("\nUnscaled Logistic Regression")
log = LogisticRegression()
log.fit(x_train, y_train)
y_pred_log = log.predict(x_test)
acc_log = accuracy_score(y_test, y_pred_log)*100
print(y_pred_lin, "\n", acc_log)

print("\nStandard Scaled Logistic Regression")
log_scaled = LogisticRegression()
log_scaled.fit(x_train_scaled, y_train_scaled)
y_pred_scaled_log = log_scaled.predict(x_test_scaled)
acc_scaled_log = accuracy_score(y_test_scaled, y_pred_scaled_log)*100
print(y_pred_scaled_log, "\n", acc_scaled_log)

print("\nMin-Max Scaled Logistic Regression")
log_min_max_scaled = LogisticRegression()
log_min_max_scaled.fit(x_train_min_max_scaled, y_train_min_max_scaled)
y_pred_min_max_scaled_log = log_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_log = accuracy_score(y_test_min_max_scaled, y_pred_min_max_scaled_log)*100
print(y_pred_min_max_scaled_log, "\n", acc_min_max_scaled_log)

print("\nUnscaled Decision Tree Classifier")
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)*100
print(y_pred_dt, "\n", acc_dt)

print("\nStandard Scaled Decision Tree Classifier")
dt_scaled = DecisionTreeClassifier()
dt_scaled.fit(x_train_scaled, y_train_scaled)
y_pred_scaled_dt = dt_scaled.predict(x_test_scaled)
acc_scaled_dt = accuracy_score(y_test_scaled, y_pred_scaled_dt)*100
print(y_pred_scaled_dt, "\n", acc_scaled_dt)

print("\nMin-Max Scaled Decision Tree Classifier")
dt_min_max_scaled = DecisionTreeClassifier()
dt_min_max_scaled.fit(x_train_min_max_scaled, y_train_min_max_scaled)
y_pred_min_max_scaled_dt = dt_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_dt = accuracy_score(y_test_min_max_scaled, y_pred_min_max_scaled_dt)*100
print(y_pred_min_max_scaled_dt, "\n", acc_min_max_scaled_dt)

print("\nUnscaled Decision Tree Regressor")
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train, y_train)
y_pred_dt_reg = dt_reg.predict(x_test)
acc_dt_reg = accuracy_score(y_test, y_pred_dt_reg)*100
print(y_pred_dt_reg, "\n", acc_dt_reg)

print("\nStandard Scaled Decision Tree Regressor")
dt_reg_scaled = DecisionTreeRegressor()
dt_reg_scaled.fit(x_train_scaled, y_train_scaled)
y_pred_scaled_dt_reg = dt_reg_scaled.predict(x_test_scaled)
acc_scaled_dt_reg = accuracy_score(y_test_scaled, y_pred_scaled_dt_reg)*100
print(y_pred_scaled_dt_reg, "\n", acc_scaled_dt_reg)

print("\nMin-Max Scaled Decision Tree Regressor")
dt_reg_min_max_scaled = DecisionTreeRegressor()
dt_reg_min_max_scaled.fit(x_train_min_max_scaled, y_train_min_max_scaled)
y_pred_min_max_scaled_dt_reg = dt_reg_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_dt_reg = accuracy_score(y_test_min_max_scaled, y_pred_min_max_scaled_dt_reg)*100
print(y_pred_min_max_scaled_dt_reg, "\n", acc_min_max_scaled_dt_reg)

print("\nUnscaled Random Forest Classifier")
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)*100
print(y_pred_rf, "\n", acc_rf)

print("\nStandard Scaled Random Forest Classifier")
rf_scaled = RandomForestClassifier()
rf_scaled.fit(x_train_scaled, y_train_scaled)
y_pred_scaled_rf = rf_scaled.predict(x_test_scaled)
acc_scaled_rf = accuracy_score(y_test_scaled, y_pred_scaled_rf)*100
print(y_pred_scaled_rf, "\n", acc_scaled_rf)

print("\nMin-Max Scaled Random Forest Classifier")
rf_min_max_scaled = RandomForestClassifier()
rf_min_max_scaled.fit(x_train_min_max_scaled, y_train_min_max_scaled)
y_pred_min_max_scaled_rf = rf_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_rf = accuracy_score(y_test_min_max_scaled, y_pred_min_max_scaled_rf)*100
print(y_pred_min_max_scaled_rf, "\n", acc_min_max_scaled_rf)

print("\nUnscaled Random Forest Regressor")
rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)
y_pred_rf_reg = rf_reg.predict(x_test)
acc_rf_reg = accuracy_score(y_test, y_pred_rf_reg)*100
print(y_pred_rf_reg, "\n", acc_rf_reg)

print("\nStandard Scaled Random Forest Regressor")
rf_reg_scaled = RandomForestRegressor()
rf_reg_scaled.fit(x_train_scaled, y_train_scaled)
y_pred_scaled_rf_reg = rf_reg_scaled.predict(x_test_scaled)
acc_scaled_rf_reg = accuracy_score(y_test_scaled, y_pred_scaled_rf_reg)*100
print(y_pred_scaled_rf_reg, "\n", acc_scaled_rf_reg)

print("\nMin-Max Scaled Random Forest Regressor")
rf_reg_min_max_scaled = RandomForestRegressor()
rf_reg_min_max_scaled.fit(x_train_min_max_scaled, y_train_min_max_scaled)
y_pred_min_max_scaled_rf_reg = rf_reg_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_rf_reg = accuracy_score(y_test_min_max_scaled, y_pred_min_max_scaled_rf_reg)*100
print(y_pred_min_max_scaled_rf_reg, "\n", acc_min_max_scaled_rf_reg)