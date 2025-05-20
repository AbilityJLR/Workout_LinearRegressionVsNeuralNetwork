# Gym Members Calories Prediction

This repository contains the implementation of a Machine Learning project for predicting calories burned during exercise using a Multi-Layer Perceptron (MLP) Neural Network built with Keras and TensorFlow.

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Model Architecture](#model-architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Files Structure](#files-structure)
* [Dependencies](#dependencies)
* [Future Work](#future-work)
* [License](#license)

## Project Overview

Exercise and calorie tracking are essential for fitness planning and weight management. This project develops an MLP Neural Network to predict calories burned based on exercise features such as duration, heart rate, body temperature, age, gender, and more.

## Dataset

The dataset used is the **Gym Members Exercise Tracking** dataset from Kaggle, containing 973 records with features:

* `User_ID`: Member identifier (dropped before modeling)
* `Gender`: Male/Female (one-hot encoded)
* `Age`, `Height`, `Weight`
* `Duration` (minutes)
* `Heart_Rate` (avg bpm)
* `Body_Temp` (°C)
* `Calories_Burned` (target)

## Methodology

1. **Data Preprocessing**

   * Remove identifier columns
   * One-Hot Encoding of `Gender`
   * Train-test split (80/20)
   * Feature scaling with `StandardScaler`
2. **Baseline Model**

   * Linear Regression for initial comparison
3. **Neural Network**

   * Build and compile an MLP with Keras
   * Train for 100 epochs with validation split
4. **Evaluation**

   * Mean Squared Error (MSE)
   * R-squared (R²)
   * Learning curves and actual vs. predicted plots

## Model Architecture

```
Input Layer: 13 features
Hidden Layer 1: 256 units, ReLU
Hidden Layer 2: 128 units, ReLU
Hidden Layer 3: 64 units, ReLU
Output Layer: 1 unit, ReLU
Optimizer: Adam (lr=0.001)
Loss: Mean Squared Error
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/gym-calories-prediction.git
   cd gym-calories-prediction
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

## Usage

Launch Jupyter Notebook and run the final project notebook:

```bash
jupyter notebook [CN340]_FinalProject.ipynb
```

Follow the notebook to preprocess data, train models, and visualize results.

## Results

* **Neural Network**: MSE = 1001.19, R² = 0.985
* **Linear Regression**: MSE = 3021.45, R² = 0.912

Learning curves indicate stable convergence without overfitting. Scatter plots show predicted vs. actual calories closely aligned.

## Files Structure

```
├── Report_CN340.pdf         # Project report
├── [CN340]_FinalProject.ipynb # Jupyter notebook
├── data/                    # Dataset files (Kaggle CSV)
├── models/                  # Saved models and scalers
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)
```

## Dependencies

* Python 3.8+
* pandas
* numpy
* scikit-learn
* tensorflow (with Keras)
* matplotlib

## Future Work

* Optimize hyperparameters with Grid/Randomized Search
* Explore deeper networks or alternative architectures
* Incorporate additional features (e.g., activity type, BMI)
* Deploy as a web or mobile application

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

*Author: Your Name — Developed for CN340 Final Project*
