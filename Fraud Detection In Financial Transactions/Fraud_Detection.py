import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load and balance the dataset
df = pd.read_csv('transaction_data.csv')

# Handle missing values
df.dropna(inplace=True)

# Balance the dataset
fraudulent_data = df[df['is_fraudulent'] == 1]
non_fraudulent_data = df[df['is_fraudulent'] == 0]
balanced_data = pd.concat([fraudulent_data, non_fraudulent_data])

# Feature Engineering
balanced_data['hour_of_day'] = pd.to_datetime(balanced_data['timestamp']).dt.hour
balanced_data['day_of_week'] = pd.to_datetime(balanced_data['timestamp']).dt.dayofweek
balanced_data['is_weekend'] = (balanced_data['day_of_week'] >= 5).astype(int)

# Visualize patterns and anomalies

# Plot 1: Average Transaction Amount by Hour of Day
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
balanced_data.groupby('hour_of_day')['transaction_amount'].mean().plot(kind='bar')
plt.xlabel('Hour of Day')
plt.ylabel('Average Transaction Amount')
plt.title('Average Transaction Amount by Hour of Day')

# Plot 2: Transaction Frequency by Day of Week
plt.subplot(2, 2, 2)
balanced_data['day_of_week'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Day of Week')
plt.ylabel('Transaction Frequency')
plt.title('Transaction Frequency by Day of Week')

# Plot 3: Transaction Amount Distribution
plt.subplot(2, 2, 3)
plt.hist(balanced_data['transaction_amount'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Transaction Amount Distribution')

# Plot 4: Fraudulent vs. Non-fraudulent Transactions
plt.subplot(2, 2, 4)
balanced_data['is_fraudulent'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.xlabel('Non-Fraudulent(0) and Fraudulent(1)')
plt.ylabel('Frequency')
plt.title('Fraudulent vs. Non-fraudulent Transactions')

plt.tight_layout()
plt.show()

# data for training
X = balanced_data[['transaction_amount', 'hour_of_day', 'day_of_week', 'is_weekend']]
y = balanced_data['is_fraudulent']

# Choose classification algorithm(Remove # for choose)

# Example: Random Forest
classifier = RandomForestClassifier()

# Example: Support Vector Machine
# classifier = SVC()

# Example: Neural Network
# classifier = MLPClassifier()

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
