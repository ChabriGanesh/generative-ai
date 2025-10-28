from flask import Flask, request, jsonify
from models import generate_synthetic_data
app = Flask(__name__)
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json['data']    
    model_type = request.json['model_type']  
    params = request.json.get('params', {})
    synthetic_data = generate_synthetic_data(data, model_type, params)
    return jsonify({"synthetic_data": synthetic_data.to_dict(orient="records")})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
