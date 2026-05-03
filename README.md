# 📊 Sales Prediction ML Dashboard (Streamlit)

An interactive machine learning web app built with **Streamlit** that allows users to:
- Upload advertising sales datasets
- Train multiple ML models
- Compare performance
- Run cross-validation
- Visualize predictions using interactive Plotly charts

---

## 🚀 Live Demo
Once deployed on Streamlit Cloud, your app will be available here:

👉 https://your-app-name.streamlit.app

---

## 📌 Features

### 📁 Data Handling
- Upload CSV dataset
- Automatic data preview
- Handles common columns:
  - TV
  - Radio
  - Newspaper
  - Sales (target)

---

### 🧠 Machine Learning Models
The app supports 3 models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

### 📊 Model Comparison
- Train multiple models simultaneously
- Compare performance metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

---

### 🔁 Cross Validation
- 5-Fold Cross Validation
- Reports:
  - Mean ± Standard Deviation
  - MAE, RMSE, R²

---

### 📈 Interactive Visualizations
- Plotly-powered graphs:
  - Actual vs Predicted scatter plots
  - Ideal prediction reference line
- Fully interactive:
  - Zoom
  - Hover tooltips
  - Pan controls

---

## 🏗️ Project Structure

sales-prediction/
│── sales_prediction.py
│── requirements.txt
│── Advertising.csv
│── README.md

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sales-ml-dashboard.git
cd sales-prediction
```

---

## ⚙️ How to run

### 1. Create virtual environment
```bash
python -m venv venv
```

### Activate environment:

#### Windows:
```bash
venv\Scripts\activate
```

#### Mac/Linux:
```bash
source venv/bin/activate
```

---

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 3. Run the app locally
```bash
streamlit run sales_prediction.py
```

Then open:
http://localhost:8501

---

## 🚀 If you want next upgrades:

a. Add professional GitHub repo structure (badges, screenshots, CI/CD)  
b. Add SHAP explainability section to README + app  
c. Add Docker deployment setup (production-level hosting)
