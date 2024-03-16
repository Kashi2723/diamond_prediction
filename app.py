from Diamond_Prediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from Diamond_Prediction.logger import logger
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@logger.catch()
@app.route('/')
def home_page():
    try :
        return render_template('index.html')
    
    except Exception as e:
        logger.exception(e)

@logger.catch()
@app.route("/predict",methods = ["GET", "POST"])
def predict_datapoints():
    try :
        if request.method == "GET":
            return render_template("form.html")
    
        else:
            data = CustomData(

                carat = float(request.form.get('carat')),
                depth = float(request.form.get('depth')),
                table = float(request.form.get('table')),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
                cut = request.form.get('cut'),
                color = request.form.get('color'),
                clarity = request.form.get('clarity')      
            )

        
            final_data = data.get_data_as_dataframe()

            predict_pipeline  = PredictPipeline()

            pred = predict_pipeline.predict(final_data)

            result = round(pred[0],2)

            return render_template('result.html',final_result = result)

    except Exception as e:
        logger.exception(e)

    
if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port= 8080)