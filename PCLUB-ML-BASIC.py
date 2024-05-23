
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from skopt.space import Real, Integer, Categorical
from sklearn.svm import SVC

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seed()

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

# Apply the preprocessing to the predictors
X_processed = preprocessor.fit_transform(X)

# # Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# # Select ratio
# ratio = 0.75
 
# total_rows = X_processed.shape[0]
# train_size = int(total_rows*ratio)
 
# # Split data into test and train
# X_train = df[0:train_size]
# X_test = df[train_size:]

# total_rows = y.shape[0]
# train_size = int(total_rows*ratio)
 
# # Split data into test and train
# y_train = df[0:train_size]
# y_test = df[train_size:]

# Define the neural network model
model = Sequential([
        Flatten(),
        Dense(128, activation='relu', input_shape=X_train.shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer
])

y_pred = model.predict(X_test,verbose=0)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=0)



def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test MAE: {test_mae}')
    print(f'Test loss: {test_loss}')

    # Predict classes
    y_pred = model.predict(X_test,verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten() 
    y_test_classes = (y_test > 0.5).astype(int)
   
    print ('Accuracy: %d' % float((np.dot(y_test_classes,y_pred_classes.T) + np.dot(1-y_test_classes,1-y_pred_classes.T))/float(y_test_classes.size)*100) + '%')
   
    # Calculate accuracy metrics
    precision = precision_score(y_test_classes, y_pred_classes)
    recall = recall_score(y_test_classes, y_pred_classes)
    f1 = f1_score(y_test_classes, y_pred_classes)
    confusion = confusion_matrix(y_test_classes, y_pred_classes)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print('Confusion Matrix:')
    print(confusion)


# Example usage
evaluate_model(model, X_test, y_test)


# # Define the search space for hyperparameters
# search_space = {
#     'optimizer': Categorical(['adam', 'sgd', 'rmsprop']),
#     'neurons': Integer(32, 256),
#     'layers': Integer(1,4)
# }

# def build_model():
#     model = Sequential([
#         Flatten(input_shape=X_train.shape),
#         Dense(128, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1, activation='sigmoid')  # Output layer
# ])

#     # Compile the model
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#     return model

# keras_regressor = KerasRegressor(build_fn=build_model, verbose=0,layers=2)


# # Create the BayesSearchCV object
# bayes_cv = BayesSearchCV(
#     estimator=keras_regressor,
#     search_spaces=search_space,
#     scoring='neg_mean_squared_error',
#     cv=3,
#     n_iter=10,
#     verbose=1,
#     random_state=42
# )

# # Fit the BayesSearchCV object to find the best hyperparameters
# bayes_cv.fit(X_train, y_train)

# # Get the best model and evaluate it
# best_model = bayes_cv.best_estimator_
# test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
# y_pred = best_model.predict(X_test).flatten()

# # Print evaluation metrics
# print(f'Best Model Test MAE: {test_mae}')
# print(f'Best Model Test Loss: {test_loss}')

# # Calculate accuracy metrics
# y_pred_classes = (y_pred > 0.5).astype(int)
# y_test_classes = (y_test > 0.5).astype(int)
# print ('Accuracy: %d' % float((np.dot(y_test_classes,y_pred_classes.T) + np.dot(1-y_test_classes,1-y_pred_classes.T))/float(y_test_classes.size)*100) + '%')
# precision = precision_score(y_test_classes, y_pred_classes)
# recall = recall_score(y_test_classes, y_pred_classes)
# f1 = f1_score(y_test_classes, y_pred_classes)
# confusion = confusion_matrix(y_test_classes, y_pred_classes)

# print(f'Best Model Precision: {precision}')
# print(f'Best Model Recall: {recall}')
# print(f'Best Model F1-score: {f1}')
# print('Best Model Confusion Matrix:')
# print(confusion)

# # Save the best model
# best_model.save('best_model.keras')

# Save the model
#model.save()
model.save('my_model.keras')



