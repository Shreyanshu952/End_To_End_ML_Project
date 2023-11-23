from src.DimondPricePrediction.pipelines.prediction_pipeline import CustomData, PredictPipeline

from flask import Flask, request, render_template


# creating an object for Flask class
app = Flask(__name__)


# creating a route for home page
@app.route("/")
def home_page():
    return render_template("index.html")

# creating a route for prediction web page
@app.route("/predict", methods=["GET", "POST"])
def predict_dimond_price():
    # condition for getting a web page consiting form to fill all the required data points for prediction
    if request.method == "GET":
        return render_template("form.html")
    
    # condition for prediction after submitting the form
    else:
        # class of custom data
        data = CustomData(
            carat=float(request.form.get("carat")),
            depth=float(request.form.get("depth")),
            table=float(request.form.get("table")),
            x=float(request.form.get("x")),
            y=float(request.form.get("y")),
            z=float(request.form.get("z")),
            cut=request.form.get("cut"),
            color=request.form.get("color"),
            clarity=request.form.get("clarity")
        )

        # converting data (filled in form) into dataframe
        final_data = data.get_data_into_dataframe()

        # class of PredictPipeline
        predict_pipeline = PredictPipeline()

        # calling predict function for prediction
        predicted_price = predict_pipeline.predict(final_data)

        # we will get predicted_price in the form of array
        # we will round it off for 2 decimal digits
        result=round(predicted_price[0], 2)

        return render_template("result.html", final_result=result)



# calling run function
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)