# Stroke Prediction Dashboard with Plotly Dash

## Project Overview
This project utilizes the stroke dataset from Kaggle to compare the effectiveness of four different classifiers: Random Forest, Logistic Regression, AdaBoost, and XGBoost. 
The main goal is to demonstrate the use of Plotly Dash for interactive data visualization and outcome prediction, allowing users to explore how different classifiers perform on the same dataset.

## Models
Logistic Regression
Random Forest
XGBoost_model
AdaBoost

## Requirements
- Python 3.8+
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- Dash
  
## Usage 
1. Clone the repository
  ```
cd Stroke
```
2. The notebook code/model.ipynb to explore or change paramteres of the ML models 
3. All models are saved under model/{model_name}.pkl
4. To start the Dash application for visualizing the dashboard
```
cd visualisation
python dashboard.py

```

## Dashboard
Explore the distribution of the dataset
![Screenshot 2024-08-03 at 4 58 40 PM](https://github.com/user-attachments/assets/c82f48d3-5153-427c-878b-c9c89dd125fc)

Prediction Model
![Screenshot 2024-08-03 at 4 58 51 PM](https://github.com/user-attachments/assets/18f67f0f-c125-43da-a05b-1becd71deb8d)
