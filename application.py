from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomDataParser, PredictPipeline

application = Flask(__name__)


@application.route("/")
def index():
    return render_template("index.html")


@application.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data_parser = CustomDataParser(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        data_df = data_parser.get_data_as_dataframe()
        print(data_df)

        pred_pipeline = PredictPipeline()
        prediction = pred_pipeline.predict(data_df)

        return render_template("home.html", results=prediction[0])


if __name__ == "__main__":
    application.run(host="0.0.0.0")
