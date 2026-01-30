import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

data = pd.read_csv('BIKE DETAILS.csv')
df = pd.DataFrame(data)

X = df[['km_driven']]
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
c = model.intercept_

print(f"Linear Regression Equation: Price = {m:.2f} × KM_Driven + {c:.2f}")

user_km = float(input("Enter kilometers driven: "))
user_km_df = pd.DataFrame({'km_driven': [user_km]})

predicted_price = model.predict(user_km_df)
print(f"Predicted bike price: {predicted_price[0]:.2f}")
