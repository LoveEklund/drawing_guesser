from flask import Flask, request, jsonify, render_template
import base64
import io
import numpy as np
from model.model import get_model
from PIL import Image
import pickle

import random

# Load dictionary using pickle
with open("model/label_mapping.pkl", 'rb') as f:
    LABEL_MAPPING = pickle.load(f)


MODEL = get_model(n_classes=len(LABEL_MAPPING))
MODEL.load_weights("model/model.h5")

app = Flask(__name__)

LABEL_INDEX = random.randint(0, len(LABEL_MAPPING)-1)
LABEL = LABEL_MAPPING[LABEL_INDEX]

def string_overlap(target, strings):
    result = ''

    for i in range(len(target)):
        overlap = False

        for s in strings:
            if i < len(s) and s[i] == target[i]:
                overlap = True
                result += target[i]
                break
        
        if not overlap:
            result += '_'
    
    return result

def score_image(image):
    resized_image = image.resize((28, 28), Image.BICUBIC)
    numpy_array = np.array(resized_image).sum(axis=2) 
    numpy_array = (numpy_array >= (numpy_array.max() * 0.3)).astype("float")

    #to see how it looks after resampling
    Image.fromarray((numpy_array * 255).astype(np.uint8)).save('predict_image.png', 'PNG', quality=95)

    predictions = MODEL.predict( numpy_array.reshape(1, 28, 28, 1))[0]
    return predictions

@app.route('/your-backend-endpoint', methods=['POST'])
def save_image():
    data = request.json
    image_data = base64.b64decode(data['image'].split(",")[1])
    in_hint = data["hint"].replace(" ","")
    image = Image.open(io.BytesIO(image_data))
    image.save('input_image.png', 'PNG', quality=95)

    predictions = score_image(image)

    for label in LABEL_MAPPING:
        print(f"p({LABEL_MAPPING[label]}) = {predictions[label]}")
    #score = predictions[GUESS_INDEX]* 100

    top_guess = predictions.argmax()
    
    if max(predictions) >= 0.5:
        if top_guess != LABEL_INDEX:
            message = f"That's not it fam, looks like {LABEL_MAPPING[top_guess]} to me, I'll update the hint for you though"
            out_hint = " ".join(string_overlap(LABEL,[LABEL_MAPPING[top_guess],in_hint]))
        else:
            message = f"yes you got it, congrats! \nit was {LABEL_MAPPING[LABEL_INDEX]} i wanted"
            out_hint = " ".join(LABEL)
    else:
        message = "what is that?"
        out_hint = " ".join(in_hint)

    return jsonify({"message": message, "newHint": out_hint})
@app.route('/', methods=['GET'])
def serve_page():
    label = LABEL_MAPPING[LABEL_INDEX] 
    
    return render_template("index.html", hint = "_ " * len(label))

if __name__ == '__main__':
    app.run(debug=True)