 🧠 ONNX Qt Predictor

A C++ desktop application that performs real-time machine learning inference using [ONNX Runtime](https://onnxruntime.ai/) and [Qt6](https://www.qt.io/). This GUI app loads a pre-trained ONNX model and a JSON scaler to make predictions based on user input.

---

## ✨ Features

- Real-time predictions with ONNX Runtime
- Easy-to-use Qt6 interface
- Normalizes input using scaler mean and scale from a JSON file
- Fast, lightweight, and cross-platform compatible

---

## 🧱 Project Structure

.
├── main.cpp # Qt GUI: creates UI, handles input/output
├── infer.cpp # Core logic for ONNX inference
├── infer.h # Header for inference functions
├── scaler.json # JSON file with mean/scale for normalization
├── model.onnx # Pre-trained ONNX model
├── lib/
│ └── onnxruntime/ # ONNX Runtime includes and library files
└── predictor # Final compiled executable (after build)

yaml
Copy
Edit

---

## 🛠 Prerequisites

- C++17 or higher
- [Qt6 development libraries](https://doc.qt.io/qt-6/gettingstarted.html)
- [ONNX Runtime C++](https://onnxruntime.ai/)
- `nlohmann/json` (header-only, download it from: https://github.com/nlohmann/json)

Place the ONNX Runtime headers and library in the `./lib/onnxruntime` folder like so:

lib/
└── onnxruntime/
├── include/
│ └── onnxruntime_cxx_api.h (etc.)
└── lib/
└── libonnxruntime.so

bash
Copy
Edit

---

## 🚧 Build Instructions

Run these commands directly from the root of your project:

```bash
# Step 1: Clean previous build artifacts
rm -f main.o infer.o predictor

# Step 2: Compile main.cpp
g++ -std=c++17 -O2 -c main.cpp -o main.o \
    -I./lib/onnxruntime/include \
    -I/usr/include/x86_64-linux-gnu/qt6/QtWidgets \
    -I/usr/include/x86_64-linux-gnu/qt6 \
    -I/usr/include/x86_64-linux-gnu/qt6/QtCore \
    -I/usr/include/x86_64-linux-gnu/qt6/QtGui \
    -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB

# Step 3: Compile infer.cpp
g++ -std=c++17 -O2 -c infer.cpp -o infer.o \
    -I./lib/onnxruntime/include \
    -I/usr/include/x86_64-linux-gnu/qt6/QtWidgets \
    -I/usr/include/x86_64-linux-gnu/qt6 \
    -I/usr/include/x86_64-linux-gnu/qt6/QtCore \
    -I/usr/include/x86_64-linux-gnu/qt6/QtGui \
    -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB

# Step 4: Link everything together
g++ -std=c++17 -O2 -o predictor main.o infer.o \
    -I./lib/onnxruntime/include \
    -L./lib/onnxruntime/lib -lonnxruntime \
    -lQt6Widgets -lQt6Gui -lQt6Core \
    -Wl,-rpath,'$ORIGIN/./lib/onnxruntime/lib'
🧪 Running the App
Make sure the following files are in the same directory:

predictor (compiled binary)

model.onnx (your ONNX model)

scaler.json (mean and scale for input normalization)

Example scaler.json:

json
Copy
Edit
{
  "mean": [100.0, 50.0, 25.0],
  "scale": [10.0, 5.0, 2.5]
}
Then run:

bash
Copy
Edit
./predictor
Input your values in the GUI, click Predict, and view the result instantly.

🧠 How It Works
Loads the ONNX model using Ort::Session.

Reads and parses scaler.json using nlohmann/json.

Scales user input using mean and scale.

Runs inference using ONNX Runtime.

Displays the prediction in a Qt GUI interface.

🧰 Customization
Modify the number of input fields directly in main.cpp.

Replace model.onnx with your own trained model.

Replace or update scaler.json with new preprocessing values.

Add error checking or validation for robustness in production.

📦 License
This project is licensed under the MIT License.

🙏 Acknowledgments
ONNX Runtime

Qt Framework

nlohmann/json
