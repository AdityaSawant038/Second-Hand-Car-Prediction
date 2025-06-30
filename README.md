
# 🚗 Used Car Price Prediction Using Neural Networks

This project predicts the resale price of second-hand cars using a neural network built in TensorFlow/Keras. The model is trained on real-world car data including attributes like year, rating, condition, top speed etc.

## 📌 Overview

- Predicts car prices using a regression neural network
- Built using Python, TensorFlow, and Pandas
- Developed and trained in Google Colab

## 🧠 Model Architecture

- Input: Normalized features from dataset
- Hidden Layers: 4 dense layers with ReLU activation
- Output: Single neuron for price (linear output)
- Loss Function: Mean Squared Error
- Optimizer: Adam

## 📊 Dataset

The dataset contains the following columns (example):
- On Road Old
- On Road New
- Years
- Rating
- Condition
- Top Speed
- Horsepower
- Torque
- Current Price

  
### 3. Run the notebook
Open `Carpredmodel.ipynb` in Jupyter or Google Colab and run all cells.

## 📦 Requirements

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
