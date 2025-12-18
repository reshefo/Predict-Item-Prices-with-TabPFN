print('Starting construction price prediction...')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor  # Changed from Classifier to Regressor
from tabpfn.constants import ModelVersion
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import time

start = time.time()
# Load your data
df = pd.read_excel('data.xlsx')  # or read_csv
# remove rows where "Price" is NaN
df = df.dropna(subset=["Price"])

# Combine relevant text columns into one string per row
# Choose which columns contain text you want to analyze
text_columns = ['Company Name', 'Discipline', 'Size', 'UoM', 'Spec', 'Unique SOR Code', 'Item Description', 'Specification']

# Create combined text (handling NaN values)
df['combined_text'] = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)

# Apply TF-IDF
vectorizer = TfidfVectorizer(max_features=500)  # Adjust max_features as needed
X = vectorizer.fit_transform(df['combined_text']).toarray()  

y = df['Price']
y = pd.to_numeric(y).astype(float)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Initialize a regressor (use TabPFN v2 for regression)
regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)

# Fit the model
print("\nTraining the model...")
regressor.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
predictions = regressor.predict(X_test)

# Add predictions to the original dataframe
df.loc[y_test.index, 'Predicted_Price'] = predictions


df.to_csv('data_with_predictions.csv', index=False)

# Calculate regression metrics
mae = mean_absolute_error(y_test, predictions)

print("\n" + "="*50)
print("REGRESSION METRICS:")
print("="*50)
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print("="*50)

end = time.time()
print(f"Time taken: {end - start:.4f} seconds")