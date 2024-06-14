import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + 1.5 * np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso regression with different alpha values
alphas = [0, 0.1, 1, 10]
plt.figure(figsize=(12, 8))

for alpha in alphas:
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lasso_reg.predict(X_test)

    # Plot the results
    plt.scatter(X_test, y_test, label='Actual Data')
    plt.plot(X_test, y_pred, label=f'Lasso Regression (alpha={alpha})')
    plt.xlabel('X-axis')
    plt.ylabel('y-axis')

    # Print the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Lasso Regression (alpha={alpha}) - Mean Squared Error: {mse:.2f}')

plt.legend()
plt.show()


 
