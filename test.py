import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv(r"/Users/shreyamhemanta/Documents/GitHub/Credit-Card-Fraud-Detection-2023-model/creditcard_2023.csv")

print(file.head())  # Check the loaded data

x = file.iloc[:, :-1]
y = file.iloc[:, -1]

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Standard Scaling
ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.transform(x_test)

# Min-Max Scaling
mm = MinMaxScaler()
x_train_min_max_scaled = mm.fit_transform(x_train)
x_test_min_max_scaled = mm.transform(x_test)

accuracies = {}

print("\nUnscaled Logistic Regression")
log = LogisticRegression()
log.fit(x_train, y_train)
y_pred_log = log.predict(x_test)
acc_log = accuracy_score(y_test, y_pred_log) * 100
accuracies['Unscaled'] = acc_log
print("Accuracy (Unscaled):", acc_log)

print("\nStandard Scaled Logistic Regression")
log_scaled = LogisticRegression()
log_scaled.fit(x_train_scaled, y_train)
y_pred_scaled_log = log_scaled.predict(x_test_scaled)
acc_scaled_log = accuracy_score(y_test, y_pred_scaled_log) * 100
accuracies['Standard Scaled'] = acc_scaled_log
print("Accuracy (Standard Scaled):", acc_scaled_log)

print("\nMin-Max Scaled Logistic Regression")
log_min_max_scaled = LogisticRegression()
log_min_max_scaled.fit(x_train_min_max_scaled, y_train)
y_pred_min_max_scaled_log = log_min_max_scaled.predict(x_test_min_max_scaled)
acc_min_max_scaled_log = accuracy_score(y_test, y_pred_min_max_scaled_log) * 100
accuracies['Min-Max Scaled'] = acc_min_max_scaled_log
print("Accuracy (Min-Max Scaled):", acc_min_max_scaled_log)



# Visualizing Logistic Regression Decision Boundary (for two features only)
if len(x.columns) == 2:
    # Logistic Regression model
    logistic_reg = LogisticRegression()
    logistic_reg.fit(x_train, y_train)

    # Creating a mesh grid of points to plot decision boundary
    x_min, x_max = x.iloc[:, 0].min() - 0.1, x.iloc[:, 0].max() + 0.1
    y_min, y_max = x.iloc[:, 1].min() - 0.1, x.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                         np.linspace(y_min, y_max, 1000))

    # Predict for each point in the mesh grid
    Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

    # Plot the data points
    plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor='k')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.title('Logistic Regression Decision Boundary')
    plt.show()
else:
    print("Visualization is available only for datasets with 2 features.")