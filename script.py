import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
db = pd.read_csv('transactions_modified.csv')
# print(transactions.head())
# print(transactions.info())

# How many fraudulent transactions?
print(db['isFraud'].value_counts()) # isFraud by amount = 282 total

# Summary statistics on amount column
print(db['amount'].describe())

# Create isPayment field
db['isPayment'] = np.where(db['type'] == 'PAYMENT', 1, 0)

# Create isMovement field
db['isMovement'] = np.where(db['type'] == 'TRANSFER', 1, 0)

# Create accountDiff field
db['accountDiff'] = np.where(db['oldbalanceOrg'] > db['newbalanceOrig'], 1, 0)

# Create features and label variables
X = db[['amount', 'isPayment', 'isMovement', 'accountDiff']]
y = db['isFraud']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Normalize the features variables
scaler = StandardScaler()
scaler.fit(x_train)

# Fit the model to the training data
model = LogisticRegression()
model.fit(x_train, y_train)

# Score the model on the training data
print(model.score(x_train, y_train))

# Score the model on the test data
print(model.score(x_test, y_test))

# Print the model coefficients
print(model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
new_transactions = np.array([transaction1, transaction2, transaction3])

# Combine new transactions into a single array
X_new = np.concatenate([X, new_transactions])

# Normalize the new transactions
X_new = scaler.transform(X_new)

# Predict fraud on the new transactions
y_pred = model.predict(X_new)

# Show probabilities on the new transactions with plt bar chart
plt.hist(y_pred)
plt.xlabel('Predicted Probability of Fraud')
plt.ylabel('Frequency')
plt.title('Predicted Probabilities')
plt.show()

