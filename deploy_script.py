from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('tesla_stock_prediction/ai_trial1.ipynb')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        input_data = np.array(data['input_data']).reshape(1, model.input_shape[1], 1)

        # Make predictions
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': float(prediction[0][0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
