from fastapi import FastAPI, File, UploadFile
from typing import List
import numpy as np
from PIL import Image
import io
import requests
import time
import json

app = FastAPI()

with open("classes.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f]

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = img.resize((512, 512))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def top_result(prob_list):
    result = np.array(prob_list)
    exp_vals = np.exp(result - np.max(result))
    softmax_out = exp_vals / np.sum(exp_vals)
    top_idx = np.argsort(softmax_out)[::-1]
    top_probs = np.round(softmax_out[top_idx] * 100, 4)
    top_classes = [classes[idx] for idx in top_idx]
    result = {}
    for i in range(3):
        result[f'Top {i+1}'] = {'name': top_classes[i], 'prob': top_probs[i]}
    return result

@app.post("/predict")
async def predict(images: List[UploadFile] = File(...)):
    res_dict = {}
    input_array = []
    for image in images:
        image_data = await image.read()
        image_array = preprocess_image(image_data)
        input_array.append(image_array)
    input_array = np.concatenate(input_array, axis=0)

    data = {
        "inputs": [
            {
                "name": "input",
                "shape": list(input_array.shape),
                "datatype": "FP32",
                "data": input_array.flatten().tolist()
            }
        ]
    }

    url_pt = "http://triton:8000/v2/models/pytorch_model/infer"
    url_pt_opt = "http://triton:8000/v2/models/pytorch_model_pruned/infer"
    url_onnx = "http://triton:8000/v2/models/onnx_model/infer"
    url_onnx_opt = "http://triton:8000/v2/models/onnx_model_opt/infer"
    url_trt = "http://triton:8000/v2/models/trt_model/infer"
    url_trt_opt = "http://triton:8000/v2/models/trt_model_fp16/infer"

    url_list = [url_pt, url_pt_opt, url_onnx, url_onnx_opt]

    tds = []
    main_results = []
    images_r = []
    for url in url_list:
        st = time.perf_counter()
        main_result = requests.post(url, data=json.dumps(data)).json()
        et = time.perf_counter()
        td = str(round(et-st, 4)) + ' сек.'
        tds += [td]
        main_results += [main_result]
        images_r += [np.array(main_result['outputs'][0]['data']).reshape(*main_result['outputs'][0]['shape'])]

    for j, image in enumerate(images):
        res_dict[image.filename] = {}
        for i in range(len(url_list)):
            image_result = main_results[i]
            top_res = top_result(images_r[i][j])
            res_dict[image.filename][image_result['model_name']] = {
                'time': tds[i],
                'result': top_res
            }

    return res_dict