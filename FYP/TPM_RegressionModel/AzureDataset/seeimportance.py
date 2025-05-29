import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Load model
loaded_model = joblib.load('rul_regressor_xgb_log_best_model.pkl')

# Get raw booster from the sklearn wrapper
booster = loaded_model.get_booster()

# Plot
xgb.plot_importance(booster, max_num_features=20)
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()
