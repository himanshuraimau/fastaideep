import torch
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

import math

# Load the CSV file
df = pd.read_csv('/home/himanshu/code/fastaideep/ch4/train.csv')
# Handle NaN values in 'Fare' column
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
# Create the 'male' column
df['Male'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# Create the 'Embarked_S', 'Embarked_M', and 'Embarked_C' columns
df['Embarked_S'] = df['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
df['Embarked_C'] = df['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
df['Pclass_1'] = df['Pclass'].apply(lambda x: 1 if x == 1 else 0)
df['Pclass_2'] = df['Pclass'].apply(lambda x: 1 if x == 2 else 0)
df['log_Fare'] = df['Fare'].apply(lambda x: round(math.log(x + 1), 2))
df['Ones'] = 1
# Save the modified DataFrame to a new CSV file
df.to_csv('/home/himanshu/code/fastaideep/ch4/train_modified.csv', index=False)


# Generate a list of 9 random numbers
random_numbers = [round(random.random(),2) for _ in range(9)]

# Create a dictionary with the specified column names and random numbers
loss_dict = {
    'Male': random_numbers[0],
    'Embarked_S': random_numbers[1],
    'Embarked_C': random_numbers[2],
    'Pclass_1': random_numbers[3],
    'Pclass_2': random_numbers[4],
    'log_Fare': random_numbers[5],
    'SibSp': random_numbers[6],
    'Parch': random_numbers[7],
    'Ones': random_numbers[8]  # Assuming 'Ones' should be 1 as in the DataFrame
}


import torch.nn as nn
import torch.optim as optim

# Load the modified CSV file
df = pd.read_csv('/home/himanshu/code/fastaideep/ch4/train_modified.csv')

# Define the feature columns and target column
feature_cols = ['Male', 'Embarked_S', 'Embarked_C', 'Pclass_1', 'Pclass_2', 'log_Fare', 'SibSp', 'Parch', 'Ones']
target_col = 'Survived'

# Handle NaN values
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

# Split the data into features and target
X = df[feature_cols].values
y = df[target_col].values

# Standardize the features manually
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1  # Prevent division by zero
X = (X - mean) / std

# Split the data into training and testing sets manually
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the linear regression model with ReLU activation
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear(x)  # Remove ReLU activation for regression

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
losses = []
epoch_plots = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]  # Epochs to plot
predictions_dict = {}  # Dictionary to store predictions for specified epochs

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Store loss for plotting
    if epoch == 0 or (epoch+1) % 100 == 0:
        losses.append((epoch, loss.item()))

    # Save predictions for specified epochs
    if epoch in epoch_plots:
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
        predictions_dict[epoch] = predictions.numpy()

# Plot the combined graph for the specified epochs
plt.figure(figsize=(10, 5))
random_indices = random.sample(range(len(predictions)), 10)
for epoch, preds in predictions_dict.items():
    plt.plot(random_indices, preds[random_indices], label=f'Epoch {epoch+1}')
plt.plot(random_indices, y_test_tensor[random_indices], 'ro', label='Actual')
plt.xlabel('Sample Index')
plt.ylabel('Survived')
plt.legend()
plt.title('Actual vs Predicted Survival at Different Epochs')
plt.savefig('/home/himanshu/code/fastaideep/ch4/actual_vs_predicted_epochs_combined.png')
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Add predictions to the DataFrame
df.loc[train_size:, 'Predictions'] = predictions.numpy()

# Save the DataFrame with predictions to a new CSV file
df.to_csv('/home/himanshu/code/fastaideep/ch4/train_with_predictions.csv', index=False)

# Plot the loss over epochs
epochs, loss_values = zip(*losses)
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_values, 'b-', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig('/home/himanshu/code/fastaideep/ch4/loss_over_epochs.png')
plt.show()

