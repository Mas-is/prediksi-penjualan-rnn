import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam


CSV_FILE = "retail_sales.csv"
INPUT_DAYS = 30
OUTPUT_DAYS = 10


def detect_column(df, candidates):
    lower_cols = {col.lower().strip(): col for col in df.columns}

    for c in candidates:
        key = c.lower().strip()
        if key in lower_cols:
            return lower_cols[key]

    return None


def prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    date_col = detect_column(df, [
        "date", "tanggal", "order date", "transaction date"
    ])

    sales_col = detect_column(df, [
        "sales", "penjualan", "total amount", "total_amount",
        "revenue", "amount", "quantity"
    ])

    if date_col is None:
        raise ValueError("Kolom tanggal tidak ditemukan. Pastikan ada kolom Date atau tanggal.")

    if sales_col is None:
        raise ValueError("Kolom penjualan tidak ditemukan. Pastikan ada kolom Total Amount, Sales, atau Penjualan.")

    df[date_col] = pd.to_datetime(df[date_col])
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])

    daily = df.groupby(date_col)[sales_col].sum().reset_index()
    daily = daily.sort_values(date_col)

    daily = daily.set_index(date_col).asfreq("D")
    daily[sales_col] = daily[sales_col].fillna(0)
    daily = daily.reset_index()

    daily.columns = ["tanggal", "penjualan"]

    return daily, date_col, sales_col


def create_sequence(data, input_days=30, output_days=10):
    X, y = [], []

    for i in range(len(data) - input_days - output_days + 1):
        X.append(data[i:i + input_days])
        y.append(data[i + input_days:i + input_days + output_days])

    return np.array(X), np.array(y)


daily, date_col, sales_col = prepare_data(CSV_FILE)

if len(daily) < INPUT_DAYS + OUTPUT_DAYS:
    raise ValueError("Data terlalu sedikit. Minimal butuh 40 hari data.")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily[["penjualan"]])

X, y = create_sequence(scaled_data, INPUT_DAYS, OUTPUT_DAYS)
y = y.reshape(y.shape[0], y.shape[1])

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

model = Sequential()

model.add(SimpleRNN(
    units=64,
    activation="tanh",
    return_sequences=True,
    input_shape=(INPUT_DAYS, 1)
))

model.add(Dropout(0.2))

model.add(SimpleRNN(
    units=32,
    activation="tanh",
    return_sequences=False
))

model.add(Dropout(0.2))

model.add(Dense(OUTPUT_DAYS))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

model.save("model_rnn_sales.keras")
joblib.dump(scaler, "scaler.pkl")

metadata = {
    "date_column": date_col,
    "sales_column": sales_col,
    "input_days": INPUT_DAYS,
    "output_days": OUTPUT_DAYS
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f)

print("Training selesai.")
print("Model disimpan sebagai model_rnn_sales.keras")
print("Scaler disimpan sebagai scaler.pkl")
print("Metadata disimpan sebagai metadata.json")