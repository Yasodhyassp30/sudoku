from flask import Flask, render_template, request, jsonify
import cv2
from io import BytesIO
from flask_cors import CORS
import numpy as np
from reader.reader import read_from_image,solve_from_image

app = Flask(__name__)
CORS(app)


def process_image(img_bytes):
    nparr = np.frombuffer(img_bytes.read(), np.uint8)
    

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    matrix, image,drawn,u_id = read_from_image(img)
    if matrix is None or image is None or drawn is None:
        return None, None, None
    _, encoded_img = cv2.imencode('.png', image)
    _, encoded_drawn = cv2.imencode('.png', drawn)
    processed_img_bytes = BytesIO(encoded_img.tobytes())
    drawn_img_bytes = BytesIO(encoded_drawn.tobytes())
    return processed_img_bytes,matrix,drawn_img_bytes,u_id

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    processed_img,matrix,drawn,u_id = process_image(file)
    if processed_img is None or matrix is None or drawn is None:
        return jsonify({'error': 'Invalid file'}), 400
    return jsonify({'processed_image': processed_img.getvalue().hex(), 'matrix': matrix, 'drawn': drawn.getvalue().hex(),'id':u_id}),200

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        matrix = data['matrix']
        u_id = data['id']
        image,output = solve_from_image(matrix,u_id)
        _, encoded_img = cv2.imencode('.png', image)
        processed_img_bytes = BytesIO(encoded_img.tobytes())
        return jsonify({'processed_image': processed_img_bytes.getvalue().hex(),'output':output}),200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Invalid matrix'}), 400

if __name__ == '__main__':
    app.run(debug=True)
