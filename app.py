from flask import Flask, request, jsonify, render_template, make_response
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

BACKUP_LABEL_INDEX = 0 


import cairocffi as cairo

def vector_to_raster(vector_image, side=28, line_diameter=12, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    function taken from here wit slight modifications 
    https://github.com/googlecreativelab/quickdraw-dataset/issues/19
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()
    
    bbox = np.hstack(vector_image).max(axis=1)
    offset = ((original_side, original_side) - bbox) / 2.
    offset = offset.reshape(-1,1)
    centered = [stroke + offset for stroke in vector_image]

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)        
    for xv, yv in centered:
        ctx.move_to(xv[0], yv[0])
        for x, y in zip(xv, yv):
            ctx.line_to(x, y)
        ctx.stroke()

    data = surface.get_data()
    raster_image = np.copy(np.asarray(data)[::4])
    
    
    return raster_image

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
    image = image / 255
    Image.fromarray((image * 255).astype(np.uint8)).save('predict_image.png', 'PNG', quality=95)
    predictions = MODEL.predict( image.reshape(1, 28, 28, 1))[0]
    return predictions

@app.route('/rate_drawing', methods=['POST'])
def save_image():
    data = request.json
    drawing_coords = data["coordinates"]
    cord_vec = []
    for stroke in drawing_coords:
        cord_vec.append(np.array([np.array([cords["x"], cords["y"]]) for cords in stroke]).T)

    image = vector_to_raster(cord_vec).reshape((28,28))
    in_hint = data["hint"].replace(" ","")


    cookie_value = request.cookies.get('label_index')
    if cookie_value:
        label_index = int(cookie_value)
    else:
        label_index = BACKUP_LABEL_INDEX
    label = LABEL_MAPPING[label_index]

    predictions = score_image(image)

    for i_label in LABEL_MAPPING:
        print(f"p({LABEL_MAPPING[i_label]}) = {predictions[i_label]}")

    top_guess = predictions.argmax()
    
    if max(predictions) >= 0.5:
        if top_guess != label_index:
            message = f"Looks like {LABEL_MAPPING[top_guess]} to me, I'll update the hint for you though"
            out_hint = " ".join(string_overlap(label,[LABEL_MAPPING[top_guess],in_hint]))
        else:
            message = f"yes you got it, congrats! \nit was {LABEL_MAPPING[label_index]} I wanted"
            out_hint = " ".join(label)
    else:
        message = "what is that?"
        out_hint = " ".join(in_hint)

    return jsonify({"message": message, "newHint": out_hint})
@app.route('/', methods=['GET'])

def serve_page():
    label_index = random.randint(0,len(LABEL_MAPPING) - 1)
    label = LABEL_MAPPING[label_index] 
    
    rendered = render_template("index.html", hint = "_ " * len(label))

    # Create a response object from the rendered string
    resp = make_response(rendered)

    # Set the cookie on the response object
    resp.set_cookie('label_index', str(label_index))
    return resp

if __name__ == '__main__':
    app.run(debug=True)