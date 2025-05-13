import joblib, json
scaler = joblib.load("scaler.pkl")
params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
with open("scaler_params.json","w") as f:
    json.dump(params, f)
