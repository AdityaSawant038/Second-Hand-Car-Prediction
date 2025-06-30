
# 🚗 Used Car Price Prediction Using Neural Networks

This project predicts the resale price of second-hand cars using a neural network built in TensorFlow/Keras. The model is trained on real-world car data including attributes like year, mileage, fuel type, transmission, etc.

## 📌 Overview

- Predicts car prices using a regression neural network
- Built using Python, TensorFlow, and Pandas
- Developed and trained in Google Colab

## 🧠 Model Architecture

- Input: Normalized features from dataset
- Hidden Layers: 1-2 dense layers with ReLU activation
- Output: Single neuron for price (linear output)
- Loss Function: Mean Squared Error
- Optimizer: Adam

## 📊 Dataset

The dataset contains the following columns (example):
- Car Name / Brand
- Year of Manufacture
- Fuel Type
- Kilometers Driven
- Seller Type
- Transmission
- Owner Type
- Selling Price

> You can use any dataset like the one from [Kaggle: Car Dekho Dataset](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho).

### 3. Run the notebook
Open `UsedCarPricePrediction.ipynb` in Jupyter or Google Colab and run all cells.

## 📦 Requirements

See `requirements.txt`, but typical packages include:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## 🧪 Results

- Achieved [insert metric: e.g. MSE = 1.23] on test data
- Visualized prediction vs actual with scatter plot

## 🔮 Future Work

- Use feature selection or PCA
- Try other models (e.g. XGBoost, SVR)
- Deploy model via Flask or Streamlit

## 🤝 Acknowledgments

- [Andrew Ng's Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- [Kaggle: CarDekho Vehicle Dataset](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)
