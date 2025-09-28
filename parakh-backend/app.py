from flask import Flask, request, jsonify
from flask_cors import CORS
import os, base64

app = Flask(__name__)
CORS(app)  # allow all origins; in production, configure specific domains:contentReference[oaicite:3]{index=3}

@app.route("/detect", methods=["POST"])
def detect_microplastics():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file to a temporary location
    upload_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(upload_path)  # Flask stores the uploaded file if save() is called:contentReference[oaicite:4]{index=4}

    # *** Placeholder for ML processing ***
    # Here you would load/instantiate your microplastics detection model and run it on upload_path.
    # Example: type_, size, output_image_path = run_model(upload_path)
    # For now, we simulate the result:
    type_ = "Synthetic Fiber"
    size = "120 µm"
    output_image_path = upload_path  # (Pretend ML did nothing and we use original image)

    # Read the (processed) image file and encode to base64
    with open(output_image_path, "rb") as img_file:
        img_bytes = img_file.read()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')

    # Cleanup: remove files to keep storage clean
    try:
        os.remove(upload_path)
        if output_image_path != upload_path:
            os.remove(output_image_path)
    except Exception as e:
        print("Cleanup error:", e)

    return jsonify({"type": type_, "size": size, "image": encoded_image})

if __name__ == "__main__":
    app.run(debug=True)
