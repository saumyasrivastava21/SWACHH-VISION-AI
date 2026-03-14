import os
import shutil
import glob
import subprocess
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from wasteDetection.pipeline.training_pipeline import TrainPipeline
from wasteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from wasteDetection.constant.application import APP_HOST, APP_PORT

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"  # Saved image name


clApp = ClientApp()

# Absolute path to your Windows-compatible YOLOv5 model
MODEL_PATH = os.path.join(
    os.getcwd(),
    "yolov5/runs/train/swachh_waste_detection2/weights/best_windows.pt"
).replace("\\", "/")

# Folder to save uploaded images (ensure it exists)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        return "Training Successful!!"
    except Exception as e:
        print("Training Error:", e)
        return Response("Training Failed", status=500)


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        print("Predict API hit!")
        data = request.json
        if 'image' not in data:
            return Response("No image found in request", status=400)

        # Save uploaded image in data folder
        image_path = os.path.join(UPLOAD_FOLDER, clApp.filename)
        decodeImage(data['image'], image_path)
        print(f"Saved input image at: {image_path}")

        yolov5_dir = os.path.join(os.getcwd(), "yolov5")

        # YOLO command using absolute path
        yolov5_cmd = (
            f'python detect.py '
            f'--weights "{MODEL_PATH}" '
            f'--img 416 '
            f'--conf 0.5 '
            f'--source "{image_path}" '
            f'--name flask_pred '
            f'--save-txt '
            f'--exist-ok'
        )
        print("Running YOLO command:", yolov5_cmd)

        process = subprocess.Popen(
            yolov5_cmd,
            shell=True,
            cwd=yolov5_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = process.communicate()
        print("YOLO Output:\n", out.decode())
        if err:
            print("YOLO Errors:\n", err.decode())

        # Get latest result image
        result_dirs = glob.glob(os.path.join(yolov5_dir, "runs/detect/flask_pred*"))
        if not result_dirs:
            return Response("YOLO did not generate output", status=500)

        result_dir = max(result_dirs, key=os.path.getmtime)
        images = glob.glob(os.path.join(result_dir, "*.jpg"))
        if not images:
            return Response("Result image not found", status=500)

        result_image = images[0]

        # Encode result image
        opencodedbase64 = encodeImageIntoBase64(result_image)
        result = {"image": opencodedbase64.decode('utf-8')}

        # Clean up runs folder
        shutil.rmtree(os.path.join(yolov5_dir, "runs"), ignore_errors=True)

        return jsonify(result)

    except Exception as e:
        print("Prediction Error:", e)
        return Response("Prediction Failed", status=500)


@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        yolov5_dir = os.path.join(os.getcwd(), "yolov5")
        yolov5_cmd = (
            f'python detect.py '
            f'--weights "{MODEL_PATH}" '
            f'--img 416 '
            f'--conf 0.5 '
            f'--source 0 '  # default camera
            f'--name flask_pred_live '
            f'--exist-ok'
        )
        print("Starting live camera detection...")
        process = subprocess.Popen(
            yolov5_cmd,
            shell=True,
            cwd=yolov5_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = process.communicate()
        print("Live YOLO Output:\n", out.decode())
        if err:
            print("Live YOLO Errors:\n", err.decode())

        shutil.rmtree(os.path.join(yolov5_dir, "runs"), ignore_errors=True)
        return "Live Camera prediction finished!"

    except Exception as e:
        print("Live Prediction Error:", e)
        return Response("Live Prediction Failed", status=500)


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=True)