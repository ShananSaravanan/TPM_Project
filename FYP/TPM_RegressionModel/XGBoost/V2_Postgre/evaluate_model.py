import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load saved test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()  # Convert to Series if single column

# Load the trained model
model = joblib.load('xgboost_rul_spc_model.pkl')

# Predict on test set
y_pred = model.predict(X_test)

# Reverse log transformation for evaluation
y_test_orig = np.expm1(y_test)  # Original RUL scale
y_pred_orig = np.expm1(y_pred)  # Original RUL scale

# Create a DataFrame for actual vs predicted
eval_df = pd.DataFrame({
    'Actual_RUL': y_test_orig,
    'Predicted_RUL': y_pred_orig,
    'Residual': y_test_orig - y_pred_orig
})

# Print evaluation summary
print("\n=== Evaluation Summary ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):.2f}")
print(f"R²: {r2_score(y_test_orig, y_pred_orig):.2f}")
print("\nResidual Statistics:")
print(eval_df['Residual'].describe())
print(f"\nProportion of Actual_RUL == 300: {(eval_df['Actual_RUL'] == 300).mean():.2%}")

# Display first 10 rows of actual vs predicted
print("\n=== Actual vs Predicted (First 10 Rows) ===")
print(eval_df.head(10))

# Save evaluation results to CSV
eval_df.to_csv('actual_vs_predicted_rul.csv', index=False)
print("\nEvaluation results saved to 'actual_vs_predicted_rul.csv'")

# Feature importance
features = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'volt_rotate', 'pressure_vibration',
            'time_since_last_maint', 'error_count']
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
print("\n=== Feature Importance ===")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Residual histogram (local plotting)
plt.figure(figsize=(8, 6))
plt.hist(eval_df['Residual'], bins=30, edgecolor='black')
plt.xlabel('Residual (Actual RUL - Predicted RUL)')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.tight_layout()
plt.savefig('residual_histogram.png')
print("\nResidual histogram saved to 'residual_histogram.png'")

# Chart.js scatter plot for actual vs predicted RUL
print("\n=== Scatter Plot: Actual vs Predicted RUL ===")
print("""```chartjs
{
  "type": "scatter",
  "data": {
    "datasets": [
      {
        "label": "Actual vs Predicted RUL",
        "data": [
""")
for i in range(len(y_test_orig)):
    print(f'          {{ "x": {y_test_orig.values[i]}, "y": {y_pred_orig[i]} }},')
print("""        ],
        "backgroundColor": "rgba(75, 192, 192, 0.6)",
        "borderColor": "rgba(75, 192, 192, 1)",
        "pointRadius": 5
      },
      {
        "label": "Ideal Line",
        "data": [
          {"x": """ + str(y_test_orig.min()) + """, "y": """ + str(y_test_orig.min()) + """},
          {"x": """ + str(y_test_orig.max()) + """, "y": """ + str(y_test_orig.max()) + """}
        ],
        "type": "line",
        "borderColor": "rgba(255, 99, 132, 1)",
        "borderWidth": 2,
        "fill": false,
        "pointRadius": 0
      }
    ]
  },
  "options": {
    "scales": {
      "x": {
        "title": {
          "display": true,
          "text": "Actual RUL (hours)"
        }
      },
      "y": {
        "title": {
          "display": true,
          "text": "Predicted RUL (hours)"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": true
      },
      "title": {
        "display": true,
        "text": "Actual vs Predicted RUL"
      }
    }
  }
}
```""")