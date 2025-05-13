#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include "infer.h"

class PredictorApp : public QWidget {
public:
    PredictorApp(QWidget* parent = nullptr)
        : QWidget(parent), env(ORT_LOGGING_LEVEL_WARNING, "Predictor") {

        setWindowTitle("Price Predictor");
        auto* layout = new QVBoxLayout(this);

        inputField = new QLineEdit(this);
        inputField->setPlaceholderText("e.g. 3,2,85,60,14.5995,120.98,17558.8,1,4166666.7,83333.3");
        layout->addWidget(inputField);

        auto* predictButton = new QPushButton("Predict", this);
        layout->addWidget(predictButton);

        resultLabel = new QLabel("Predicted Price: ", this);
        layout->addWidget(resultLabel);

        // Load scaler params
        try {
            infer::load_scaler("scaler_params.json", scalerMean, scalerScale);
        } catch (const std::exception& e) {
            showError("Error loading scaler parameters: " + QString::fromStdString(e.what()));
            return;
        }

        // Load ONNX model
        try {
            onnxSession = infer::loadModel("mlp_model.onnx", env);
        } catch (const std::exception& e) {
            showError("Error loading ONNX model: " + QString::fromStdString(e.what()));
            return;
        }

        // Click handler
        connect(predictButton, &QPushButton::clicked, this, [=]() {
            QStringList parts = inputField->text().split(',');
            std::vector<float> vals;
            for (const auto& p : parts) {
                if (!p.trimmed().isEmpty()) {
                    bool ok;
                    float val = p.trimmed().toFloat(&ok);
                    if (ok) {
                        vals.push_back(val);
                    } else {
                        showError("Invalid input format. Please enter valid numbers.");
                        return;
                    }
                }
            }

            if (vals.size() != scalerMean.size()) {
                showError("Error: Input count must match scaler size.");
                return;
            }

            std::vector<int64_t> inShape = {1, static_cast<int64_t>(vals.size())};
            std::vector<int64_t> outShape = {1};

            try {
                float price = infer::predict(vals, scalerMean, scalerScale, *onnxSession);
                resultLabel->setText("Predicted Price: ₱" + QString::number(price));
            } catch (const std::exception& e) {
                showError("Prediction error: " + QString::fromStdString(e.what()));
            }
        });
    }

private:
    QLineEdit* inputField;
    QLabel* resultLabel;
    Ort::Env env;
    std::unique_ptr<Ort::Session> onnxSession;
    std::vector<float> scalerMean, scalerScale;

    void showError(const QString& message) {
        QMessageBox::critical(this, "Error", message);
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    PredictorApp w;
    w.show();
    return app.exec();
}
