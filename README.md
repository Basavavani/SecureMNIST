# SecureMNIST
Secure Federated Learning Project using MNIST dataset - B.Tech Data Science
# 🔐 Secure Federated Learning on MNIST Dataset

![GitHub repo size](https://img.shields.io/github/repo-size/Basavavani/secureMNIST)
![GitHub stars](https://img.shields.io/github/stars/Basavavani/secureMNIST?style=social)
![GitHub forks](https://img.shields.io/github/forks/Basavavani/secureMNIST?style=social)
![Last Commit](https://img.shields.io/github/last-commit/Basavavani/secureMNIST)
![License](https://img.shields.io/github/license/Basavavani/secureMNIST)

A privacy-preserving machine learning project that simulates **Federated Learning** using the **MNIST dataset** across multiple virtual clients without sharing raw data.

---

## 📌 Project Overview

This project demonstrates how federated learning works by training local models on two separate datasets (simulating two clients) and combining their updates using **Federated Averaging** (FedAvg) — all while ensuring data privacy.

---

## 🚀 Features

- Federated training using two clients
- Local training with private datasets
- Federated Averaging algorithm
- Accuracy & loss reporting
- Trained model saved as `global_model.pth`

---

## 🧰 Tech Stack

- Python 3.9  
- PyTorch  
- Torchvision  
- Visual Studio Code  

---

## 🗂️ Project Structure

```
secureMNIST/
├── train.py                 # Main training script
├── global_model.pth         # Trained global model
├── data/                    # MNIST dataset
└── README.md                # Project documentation
```

---

## 🧪 How It Works

1. Load and split the MNIST dataset into two equal parts.
2. Each client trains a local model on its data.
3. Use **FedAvg** to average weights and build a global model.
4. Evaluate the global model’s performance.

---

## 🧑‍💻 Usage

### ✅ Requirements

```bash
pip install torch==1.10.0+cpu torchvision==0.11.1+cpu
```

### ▶️ Run the Training

```bash
python train.py
```

### 📦 Output Example

```
🔒 Starting Secure Federated MNIST Training...
🏋️ Training on Client 1...
🏋️ Training on Client 2...
🔗 Performing Federated Averaging...
✅ Federated Averaging Done!
🎉 Secure Federated Training Completed.
✅ Accuracy: 72.16%
📉 Average Loss: 1.5850
💾 Global model saved as 'global_model.pth'
```

---

## 📸 Screenshots

> _You can include VS Code training output, data split view, or loss graphs here._

---

## 🔐 Real-Life Use Case (Healthcare)

Hospitals in different cities want to collaborate on building a diabetes detection model, but cannot share patient data due to privacy laws.

✅ With **Federated Learning**, each hospital trains a model locally and shares only model updates — not sensitive data. These are averaged to build a global AI model collaboratively.

---

## 📈 Future Improvements

- Extend to multiple clients
- Add secure aggregation with PySyft or CrypTen
- Visualize training accuracy in real-time
- Apply on medical, finance, or industrial datasets

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

> Built with ❤️ by [Basavavani](https://github.com/Basavavani)
