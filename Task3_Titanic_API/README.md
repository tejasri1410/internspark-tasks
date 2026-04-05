# Titanic Survival Prediction API
This project builds a Machine Learning model to predict whether a passenger survived the Titanic disaster and exposes it as a REST API using Flask.
## Features
- RandomForest ML model
- Flask API endpoint
- Docker container support

## API Endpoint
POST /predict

### Sample Request
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 25,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

### Sample Response
{
  "Survived": 0,
  "Survival_Probability": 0.03
}

## Run Locally
1. Train model:
   python train.py

2. Start API:
   python app.py

3. Test API:
   Use Postman or curl

## Run with Docker
Build image:
docker build -t titanic-api .

Run container:
docker run -p 5000:5000 titanic-api

## Note
If you face dependency issues, please install required packages using:
pip install -r requirements.txt
