import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # For saving and loading the model

# Load and prepare the dataset
file_path = r"C:\Users\Dell\Documents\House Price India.csv"
data = pd.read_csv(file_path)

# Rename columns for easier access (optional but recommended)
data.rename(columns={
    'living area': 'LivingArea',
    'number of bedrooms': 'Bedrooms',
    'Price': 'Price'
}, inplace=True)

# Ensure the required columns exist
required_columns = ['LivingArea', 'Bedrooms', 'Price']
if not all(col in data.columns for col in required_columns):
    raise KeyError(f"Required columns {required_columns} are missing in the dataset. Available columns: {data.columns}")

data = data[required_columns]

# Handle missing values
data['LivingArea'].fillna(data['LivingArea'].median(), inplace=True)
data['Bedrooms'].fillna(data['Bedrooms'].mode()[0], inplace=True)
data['Price'].fillna(data['Price'].median(), inplace=True)

# Features and target
X = data[['LivingArea', 'Bedrooms']]
y = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display metrics for developer reference
print(f"Model Evaluation Metrics:\nMean Absolute Error: {mae}\nRoot Mean Squared Error: {rmse}")

# Save the trained model
model_path = "house_price_model.pkl"
joblib.dump(model, model_path)

# Client Interaction
print("\n=== Predict House Prices ===")
while True:
    try:
        # Input from client
        living_area = float(input("Enter the living area (in square feet): "))
        bedrooms = int(input("Enter the number of bedrooms: "))

        # Prepare input data for prediction
        input_data = pd.DataFrame({'LivingArea': [living_area], 'Bedrooms': [bedrooms]})

        # Load model and predict
        loaded_model = joblib.load(model_path)
        predicted_price = loaded_model.predict(input_data)
        print(f"\nThe predicted house price is: ₹{predicted_price[0]:,.2f}")

        # Option to predict again
        another = input("\nDo you want to predict another house price? (yes/no): ").lower()
        if another != 'yes':
            print("Thank you for using the house price prediction tool!")
            break
    except Exception as e:
        print(f"Error: {e}. Please try again.")
