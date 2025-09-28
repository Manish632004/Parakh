from flask import Flask, request, jsonify
from flask_cors import CORS
import os, base64
from model import analyze_image

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

    # Run model inference
    try:
        result, vis_path = analyze_image(upload_path)
    except Exception as e:
        # Cleanup temp file
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500

    # Choose an image to return: visualization if available, else original
    output_image_path = vis_path if (vis_path and os.path.exists(vis_path)) else upload_path

    with open(output_image_path, "rb") as img_file:
        img_bytes = img_file.read()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')

    # Derive simple summary fields for backward-compat
    particles = result.get("particles", [])
    if particles:
        # Majority polymer type
        from collections import Counter
        majority_type = Counter([p.get("polymer_type", "unknown") for p in particles]).most_common(1)[0][0]
        # Average size in microns
        avg_size = sum(p.get("size_microns", 0) for p in particles) / max(len(particles), 1)
        type_ = majority_type
        size = f"{avg_size:.0f} µm"
    else:
        type_ = "none"
        size = "0 µm"

    # Cleanup: remove files to keep storage clean
    try:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if output_image_path != upload_path and os.path.exists(output_image_path):
            os.remove(output_image_path)
    except Exception as e:
        print("Cleanup error:", e)

    return jsonify({
        "type": type_,
        "size": size,
        "image": encoded_image,
        "result": result
    })

if __name__ == "__main__":
    app.run(debug=True)
