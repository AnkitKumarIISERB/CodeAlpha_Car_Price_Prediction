from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("best_car_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get form inputs
            year = int(request.form["year"])
            present_price = float(request.form["present_price"])
            driven_kms = int(request.form["driven_kms"])
            brand = request.form["brand"]
            fuel_type = request.form["fuel_type"]
            selling_type = request.form["selling_type"]
            transmission = request.form["transmission"]
            owner = int(request.form["owner"])

            # Calculate car age
            current_year = datetime.now().year
            car_age = current_year - year

            # Create dataframe with EXACT training column names
            input_df = pd.DataFrame([{
                "Present_Price": present_price,
                "Driven_kms": driven_kms,
                "Owner": owner,
                "Car_Age": car_age,
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
                "Brand": brand
            }])

            # Predict
            pred = model.predict(input_df)[0]
            prediction = round(pred, 2)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
