import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import joblib

# Load the data
df = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')  # Make sure to adjust the path accordingly

# Load models
model_names = ['Random Forest', 'Logistic Regression', 'AdaBoost', 'XGBoost']
models = {name: joblib.load(f"../model/{name}_model.pkl") for name in model_names}

app = dash.Dash(__name__)

# Define dropdown options for various categorical features
gender_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}]
bmi_options = [{'label': 'Obesity', 'value': 'Obesity'}, {'label': 'Overweight', 'value': 'Overweight'}, {'label': 'Underweight', 'value': 'Underweight'}, {'label': 'Normal weight', 'value': 'Normal weight'}]
hypertension_options = [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]
heart_disease_options = [{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}]
marital_status_options = [{'label': 'Married', 'value': 'Yes'}, {'label': 'Not Married', 'value': 'No'}]
glucose_cat_options = [{'label': 'Prediabetes', 'value': 'Prediabetes'}, {'label': 'Normal', 'value': 'Normal'},{'label': 'Diabetes', 'value': 'Diabetes'}]
work_type_options = [{'label': 'Private', 'value': 'Private'}, {'label': 'Self-employed', 'value': 'Self-employed'}, {'label': 'Government Job', 'value': 'Govt_job'}, {'label': 'Children', 'value': 'Children'}, {'label': 'Never Worked', 'value': 'Never_worked'}]
residence_type_options = [{'label': 'Urban', 'value': 'Urban'}, {'label': 'Rural', 'value': 'Rural'}]
smoking_status_options = [{'label': 'Smokes', 'value': 'smokes'}, {'label': 'Formerly Smoked', 'value': 'formerly smoked'}, {'label': 'Never Smoked', 'value': 'never smoked'}]

# Dropdown for model selection
model_options = [{'label': name, 'value': name} for name in models.keys()]

# App layout
app.layout = html.Div(style={'backgroundColor': 'white', 'padding': '20px',  'width': '100%', 'height': '70vh'}, children=[
    html.H1('Stroke Data Dashboard', style={'textAlign': 'center'}),

    html.Hr(style={'borderWidth': "0.25vh", "width": "100%", "borderColor": "#AB87FF","opacity": "unset"}),

    html.H2('Data  Distribution', style={'textAlign': 'center'}),
    
    # Dashboard for Data Distribution
    html.Div([
        html.H3('Total Patients: {}'.format(len(df)), style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='feature',
            options=[
                {'label': 'Gender', 'value': 'gender'},
                {'label': 'Hypertension', 'value': 'hypertension'},
                {'label': 'Heart Disease', 'value': 'heart_disease'},
                {'label': 'Ever Married', 'value': 'ever_married'},
                {'label': 'Work Type', 'value': 'work_type'},
                {'label': 'Residence Type', 'value': 'Residence_type'},
                {'label': 'BMI', 'value': 'bmi'},
                {'label': 'Smoking Status', 'value': 'smoking_status'}
            ],
            value='hypertension',
            clearable=False,
            style={ 'margin': '10px'}
        ),
        dcc.Graph(id='dynamic-graph'),
    ], style={'padding': '20px',  'width': '100%', 'height': '75vh'}),

    html.Hr(style={'borderWidth': "0.05vh", "width": "100%", "borderColor": "#FF0000"}),
    # Prediction Model Form
    html.Div([
        html.H2('Stroke Prediction Model', style={'textAlign': 'center'}),
        dcc.Dropdown(id='model-selector', options=model_options, value='Random Forest', placeholder='Select a Model', style={'margin': '10px'}),
        dcc.Input(id='age-input', type='number', placeholder='Enter Age', style={'margin': '20px'}),
        dcc.Dropdown(id='bmi-dropdown', options=bmi_options, placeholder='Select BMI Category', style={'margin': '10px'}),
        dcc.Dropdown(id='gender-dropdown', options=gender_options, placeholder='Select Gender', style={'margin': '10px'}),
        dcc.Dropdown(id='hypertension-dropdown', options=hypertension_options, placeholder='Select Hypertension Status', style={'margin': '10px'}),
        dcc.Dropdown(id='heart-disease-dropdown', options=heart_disease_options, placeholder='Select Heart Disease Status', style={'margin': '10px'}),
        dcc.Dropdown(id='marital-status-dropdown', options=marital_status_options, placeholder='Select Marital Status', style={'margin': '10px'}),
        dcc.Dropdown(id='glucose-cat-dropdown', options=glucose_cat_options, placeholder='Select Glucose Category', style={'margin': '10px'}),
        dcc.Dropdown(id='work-type-dropdown', options=work_type_options, placeholder='Select Work Type', style={'margin': '10px'}),
        dcc.Dropdown(id='residence-type-dropdown', options=residence_type_options, placeholder='Select Residence Type', style={'margin': '10px'}),
        dcc.Dropdown(id='smoking-status-dropdown', options=smoking_status_options, placeholder='Select Smoking Status', style={'margin': '10px'}),
        html.Button('Predict Stroke', id='predict-button', n_clicks=0, style={'margin': '20px'}),
        html.Div(id='prediction-output', style={'margin': '20px', 'textAlign': 'center'})
    ], style={'backgroundColor': 'white', 'padding': '20px',  'width': '100%', 'height': '70vh'}),
])

# Callback to update the feature distribution graph
@app.callback(
    Output('dynamic-graph', 'figure'),
    [Input('feature', 'value')]
)
def update_graph(feature):
    fig = px.histogram(df, x=feature, color='stroke', title='Stroke Distribution by {}'.format(feature.capitalize()), barmode='group')

    fig.update_layout(
    autosize=False,
    width=1500,  # Set the width to 800 pixels
    height=600,
)
    return fig

# Callback to handle the prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('model-selector', 'value'),
    State('age-input', 'value'),
    State('bmi-dropdown', 'value'),
    State('gender-dropdown', 'value'),
    State('hypertension-dropdown', 'value'),
    State('heart-disease-dropdown', 'value'),
    State('marital-status-dropdown', 'value'),
    State('glucose-cat-dropdown', 'value'),
    State('work-type-dropdown', 'value'),
    State('residence-type-dropdown', 'value'),
    State('smoking-status-dropdown', 'value')
)
def predict_stroke(n_clicks, selected_model, age, bmi_cat, gender, hypertension, heart_disease, married, glucose_cat, work_type, residence_type, smoking_status):
    if n_clicks > 0:
        input_data = pd.DataFrame([{
            'age': age, 
            'bmi_cat': bmi_cat, 
            'gender': gender, 
            'hypertension': hypertension, 
            'heart_disease': heart_disease, 
            'ever_married': married, 
            'work_type': work_type, 
            'Residence_type': residence_type, 
            'smoking_status': smoking_status,
            'glucose_cat': glucose_cat
        }])
        prediction = models[selected_model].predict(input_data)
        prediction_text = 'Stroke' if prediction[0] == 1 else 'No Stroke'
        return html.Span(f'Prediction: {prediction_text}', style={'font-weight': 'bold', 'font-size': '24px'})

if __name__ == '__main__':
    app.run_server(debug=True)
