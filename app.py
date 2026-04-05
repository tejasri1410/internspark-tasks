from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('titanic_model.pkl')
le_sex = joblib.load('le_sex.pkl')
le_embarked = joblib.load('le_embarked.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([{
            "Pclass": int(data["Pclass"]),
            "Sex": le_sex.transform([data["Sex"].lower()])[0],
            "Age": float(data["Age"]),
            "SibSp": int(data["SibSp"]),
            "Parch": int(data["Parch"]),
            "Fare": float(data["Fare"]),
            "Embarked": le_embarked.transform([data["Embarked"].upper()])[0]
        }])

        survived = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        return jsonify({
            "Survived": survived,
            "Survival_Probability": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)