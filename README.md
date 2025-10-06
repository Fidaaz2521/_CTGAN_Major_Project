# 🩺 Multimorbidity Risk Prediction using CTGAN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end machine learning project that predicts multimorbidity risk in patients using advanced data augmentation techniques with Conditional Tabular Generative Adversarial Networks (CTGAN).

## 🎯 Project Overview

This project addresses the critical healthcare challenge of predicting patients at risk for multiple chronic conditions (multimorbidity). By combining statistical hypothesis testing, advanced machine learning, and synthetic data generation, we developed a robust predictive system achieving **98.5% accuracy** and **96% recall** for high-risk patients.

### Key Features
- **Statistical validation** of multimorbidity associations
- **CTGAN-based synthetic data augmentation** to address class imbalance
- **Multiple ML model comparison** (Random Forest, XGBoost, Neural Networks)
- **Interactive web dashboard** for real-time predictions
- **Batch processing capabilities** for healthcare providers

## 📊 Results Summary

| Metric | Before CTGAN | After CTGAN |
|--------|--------------|-------------|
| Overall Accuracy | 93.8% | 98.5% |
| Recall (High-risk) | 23% | 96% |
| Precision (High-risk) | 27% | 94% |
| ROC-AUC | 0.80 | 0.998 |

## 🔬 Methodology

### 1. Exploratory Data Analysis & Hypothesis Testing
- **Pearson correlation** analysis between conditions and medications (r=0.46)
- **Chi-square tests** for gender-multimorbidity associations (χ²=11.335, p=0.0008)
- **T-tests** confirming cost differences between patient groups

### 2. Model Development
- Baseline models: Random Forest, XGBoost
- Advanced neural network with class weighting
- Preprocessing pipelines with scaling and encoding

### 3. CTGAN Data Augmentation
- Generated 200 realistic synthetic multimorbid patient samples
- Preserved original data distributions
- Addressed severe class imbalance (13 vs 280 patients)

### 4. Deployment
- Streamlit web application with interactive UI
- Ngrok tunneling for public access
- Batch prediction capabilities with CSV upload/download

## 📁 Project Structure

```
├── notebooks/
│   ├── CTGAN_MajorProjectEDA.ipynb          # Exploratory Data Analysis
│   ├── CTGAN_MajorProject_HypothesisT_Modelling.ipynb  # Model Development
│   └── CTGAN_MajorProject_Deployment.ipynb  # Deployment Setup
├── models/
│   ├── best_nn_model.h5                     # Trained Neural Network
│   ├── best_nn_model_joblib.pkl            # Alternative Model Format
│   └── scaler.pkl                          # Feature Scaler
├── data/
│   └── SYNTHETIC PATIENT REPOSITORY BY SYNTHEA                 # Dataset (Healthcare Analytics)
├── app/
│   └── app.py                              # Streamlit Web Application
├── requirements.txt                        # Python Dependencies
└── README.md                              # Project Documentation
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimorbidity-ctgan-prediction.git
cd multimorbidity-ctgan-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Local Streamlit App:**
```bash
streamlit run app/app.py
```

2. **With Ngrok (Public Access):**
```python
import os
from pyngrok import ngrok
import time

os.system("nohup streamlit run app/app.py &")
time.sleep(5)
public_url = ngrok.connect(8501)
print("Open the UI Web App here:", public_url)
```

### Usage

**Single Patient Prediction:**
- Enter patient demographics and health metrics
- Get instant risk assessment with probability score
- View results in styled alerts and interactive gauge charts

**Batch Predictions:**
- Upload CSV file with patient data
- Download results with risk scores and classifications
- Process multiple patients simultaneously

## 🛠️ Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
tensorflow>=2.10.0
plotly>=5.0.0
ctgan>=0.7.0
pyngrok>=5.0.0
joblib>=1.3.0
```

## 📈 Model Performance

### Before CTGAN Augmentation
- High overall accuracy but poor minority class detection
- Random Forest: 93.8% accuracy, 23% recall for high-risk patients
- XGBoost: 91.8% accuracy, 23% recall for high-risk patients

### After CTGAN Augmentation
- Neural Network: **98.5% accuracy, 96% recall** for high-risk patients
- ROC-AUC: **0.998** (near-perfect discrimination)
- Balanced performance across both patient classes

## 🎯 Clinical Impact

This system enables healthcare providers to:
- **Identify high-risk patients** before complications develop
- **Allocate resources efficiently** based on risk stratification  
- **Implement preventive interventions** for multimorbid patients
- **Reduce healthcare costs** through early detection

## 🔮 Future Enhancements

- **Real-time EHR integration** for automated data ingestion
- **Explainability modules** (SHAP, LIME) for prediction interpretation
- **Cloud deployment** with scalable infrastructure (AWS/GCP/Azure)
- **Longitudinal risk modeling** with time-series data
- **Multi-modal inputs** (lab results, imaging, genomics)
- **Automated model monitoring** and drift detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

- **Author:** Fida Taneem
- **Email:** fidaz.datascientist02@gmail.com

- **Project Link:** https://github.com/Fidaaz2521/_CTGAN_Major_Project

## 🙏 Acknowledgments

- Healthcare dataset providers
- CTGAN developers for advanced synthetic data generation
- Open-source ML community for foundational tools
- Streamlit team for accessible web app framework

---

*This project demonstrates the power of combining traditional statistical methods with modern ML techniques and synthetic data generation to solve real-world healthcare challenges.*
