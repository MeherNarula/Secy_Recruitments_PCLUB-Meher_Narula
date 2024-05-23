import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the CSV file
df = pd.read_csv('lungcapacity.csv', encoding="utf-8")

# Rename the specific column 'Age( years)' to 'Age'
df.rename(columns={'Age( years)': 'Age'}, inplace=True)
df.rename(columns={ 'Height(inches)': 'Height'}, inplace=True)

# Fill missing values for numeric columns with the mean of the column
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Fill missing values for categorical columns with the mode of the column
categorical_columns = df.select_dtypes(include='object').columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Define the function to calculate tidal volume for men
def tidal_volume_men(row):
    age_cubed = row['Age'] ** 3
    return (23.79 + 0.00002066 * age_cubed) * row['LungCap(cc)'] / 100

# Define the function to calculate tidal volume for women
def tidal_volume_women(row):
    age_cubed = row['Age'] ** 3
    return (27.091 + 0.00002554 * age_cubed) * row['LungCap(cc)'] / 100

# Apply the tidal volume calculation based on gender
df['tidalvolume'] = df.apply(lambda row: tidal_volume_men(row) if row['Gender'] == 'M' else tidal_volume_women(row), axis=1)

# Sort the DataFrame by the 'Age' column in ascending order
df.sort_values(by='Age', ascending=True, inplace=True)

# Save the cleaned and modified DataFrame back to a new CSV file
df.to_csv('cleaned_sorted_data.csv', index=False)

# Print the head of the DataFrame
print("Head of the cleaned and sorted DataFrame:")
print(df.head())

print("Data cleaning, filling missing values, and sorting completed. The result is saved in 'cleaned_sorted_data.csv'.")

# Load your data
data = pd.read_csv('cleaned_sorted_data.csv', encoding="utf-8")

# Define predictor and target variables
X = data[['Age', 'Gender', 'Height', 'Smoke', 'Caesarean']]  # Add or remove relevant symptoms as needed
y = data['tidalvolume']

# Preprocess the data
numeric_features = ['Age', 'Height']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_features = ['Gender', 'Smoke', 'Caesarean']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])



# # Apply the preprocessing to the predictors
X_processed = preprocessor.fit_transform(X)
# y_processed = y
# print(X_processed)
# # Split the data

# train_sizeX= int(0.8 * len(X_processed))
# test_sizeX = len(X_processed) - train_sizeX
# train_sizey= int(0.8 * len(y_processed))
# test_sizey = len(y_processed) - train_sizey
# print(train_sizeX)
# print(test_sizeX)
# print(X)

# train_datasetX, test_datasetX = torch.utils.data.random_split(X_processed, [train_sizeX, test_sizeX])
# train_datasety, test_datasety = torch.utils.data.random_split(X_processed, [train_sizey, test_sizey])

# # Assuming train_datasetX, test_datasetX, train_datasety, and test_datasety are lists of tensors containing strings
# X_train_list = [train_datasetX[i][0] for i in range(len(train_datasetX))]
# X_test_list = [test_datasetX[i][0] for i in range(len(test_datasetX))]
# y_train_list = [train_datasety[i][0] for i in range(len(train_datasety))]
# y_test_list = [test_datasety[i][0] for i in range(len(test_datasety))]

# # Concatenate the lists of strings into single lists
# X_train = [item for sublist in X_train_list for item in sublist]
# X_test = [item for sublist in X_test_list for item in sublist]
# y_train = [item for sublist in y_train_list for item in sublist]
# y_test= [item for sublist in y_test_list for item in sublist]



# Split the data into train and test sets using DataLoader
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Extract data and targets from the DataLoader datasets
X_train = torch.cat([data[0] for data in train_loader.dataset], dim=0)
y_train = torch.cat([data[1] for data in train_loader.dataset], dim=0)
X_test = torch.cat([data[0] for data in test_loader.dataset], dim=0)
y_test = torch.cat([data[1] for data in test_loader.dataset], dim=0)


# Define the neural network model
model = Sequential([
    Flatten(),
    Dense(128, activation='relu', input_shape=X_train.shape),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer
])
y_pred = model.predict(y_test)

keras.config.disable_traceback_filtering() 
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(train_loader, epochs=100, validation_data=test_loader)

# Evaluate the model
test_loss, test_mae = model.evaluate(test_loader)
print(f'Test MAE: {test_mae}')
print(f'Test loss: {test_loss}')

# Save the model
model.save('my_model.keras')

