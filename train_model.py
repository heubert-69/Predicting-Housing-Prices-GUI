import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib

# Load data
df = pd.read_csv("PH_houses_v2.csv")
print(f"Initial rows: {len(df)}")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Clean and convert 'Price (PHP)'
df["Price (PHP)"] = df["Price (PHP)"].astype(str).str.replace("₱", "", regex=False)
df["Price (PHP)"] = df["Price (PHP)"].str.replace(",", "", regex=False)
df["Price (PHP)"] = pd.to_numeric(df["Price (PHP)"], errors='coerce')

# Drop rows with missing target and essential features
essential_cols = ["Price (PHP)", "Floor_area (sqm)", "Bedrooms"]
df.dropna(subset=essential_cols, inplace=True)

# Convert numeric columns
df["Floor_area (sqm)"] = pd.to_numeric(df["Floor_area (sqm)"], errors='coerce')
df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors='coerce')
df["Bath"] = pd.to_numeric(df["Bath"], errors='coerce')
df["Land_area (sqm)"] = pd.to_numeric(df["Land_area (sqm)"], errors='coerce')

# Fill remaining non-essential numeric missing values with median
df["Bath"].fillna(df["Bath"].median(), inplace=True)
df["Land_area (sqm)"].fillna(df["Land_area (sqm)"].median(), inplace=True)

print(f"After cleaning: {len(df)}")

# Convert 'Price (PHP)' to millions
df['Price_millions'] = df['Price (PHP)'] / 1e6
df.drop(columns=['Price (PHP)'], inplace=True)

# Remove outliers on numeric features
def remove_outliers(series):
    q_low = series.quantile(0.01)
    q_high = series.quantile(0.99)
    return series.between(q_low, q_high)

for col in ['Floor_area (sqm)', 'Land_area (sqm)', 'Bedrooms', 'Bath']:
    if col in df.columns:
        mask = remove_outliers(df[col])
        df = df[mask]

# Encode categorical features
df = pd.get_dummies(df)

# Separate target and features
y = df['Price_millions']
X = df.drop(columns=['Price_millions'])
print(f"After feature engineering: {len(df)}")

# Train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
joblib.dump(scaler, 'scaler.pkl')

# Define Optuna objective
def objective(trial):
    model = keras.Sequential()
    n_layers = trial.suggest_int("n_layers", 1, 3)
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
        model.add(layers.Dense(num_hidden, activation='relu'))
    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    )

    model.compile(optimizer=optimizer, loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=trial.suggest_int("batch_size", 16, 128, step=16),
        callbacks=[early_stop],
        verbose=0
    )

    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial)

# Train final model with best params
best_params = study.best_trial.params
final_model = keras.Sequential()
for i in range(best_params["n_layers"]):
    final_model.add(layers.Dense(best_params[f"n_units_l{i}"], activation='relu'))
final_model.add(layers.Dense(1))

final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
    loss='mse'
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
final_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=best_params["batch_size"],
    validation_data=(X_valid, y_valid),
    callbacks=[early_stop],
    verbose=1
)

# Save the model
final_model.save("mlp_model.keras")
print("Model training complete and saved.")
