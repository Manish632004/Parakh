import os
import tempfile
import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = 'datasets'
CSV_FILE = 'samples_geocoded.csv'  
IMG_SIZE_FOR_FEATURES = (100, 100)  # For patch processing
INPUT_IMAGE_PATH = 'sample_input_image.jpg'  # Replace with your water sample image (JPG/PNG)
PIXEL_TO_MICRON = 1.0  # Microns per pixel; set from your imaging setup (e.g., 0.5 means 0.5 µm per pixel)
MIN_AREA_PIXELS = 50  # Minimum particle area to detect
CLASSES = ['non_plastic', 'PE', 'PP', 'PS', 'PET']  # Common polymer types; adjust based on CSV
MODEL_PATH = 'microplastics_tabular_model.pkl'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, CSV_FILE)

# Step 1: Load and Preprocess Tabular Dataset
def load_and_preprocess_dataset(csv_path):
    """
    Loads the CSV dataset, preprocesses features and targets for ML.
    Assumes columns: 'size_microns', 'aspect_ratio', 'roundness', 'is_microplastic', 'polymer_type'.
    If columns differ, map them (e.g., via dict). Handles missing values.
    """
    if not os.path.exists(csv_path):
        # For demo: Create a sample CSV based on typical microplastics data if not found
        logger.warning(f"CSV not found at {csv_path}. Creating sample data for demo.")
        np.random.seed(42)
        n_samples = 1000
        data = {
            'size_microns': np.random.uniform(1, 5000, n_samples),
            'aspect_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'roundness': np.random.uniform(0.1, 1.0, n_samples),  # 1=perfect circle
            'is_microplastic': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'polymer_type': np.random.choice(CLASSES, n_samples)
        }
        # Simulate correlation: Larger sizes more likely PE/PP
        data['polymer_type'] = np.where(data['size_microns'] > 100, np.random.choice(['PE', 'PP'], n_samples, p=[0.5, 0.5]), 
                                        np.random.choice(['non_plastic', 'PS'], n_samples, p=[0.6, 0.4]))
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logger.info("Sample CSV created. In production, download real data from the dataset URL.")
    else:
        df = pd.read_csv(csv_path)
    
    # Feature columns (adjust based on actual CSV)
    feature_cols = ['size_microns', 'aspect_ratio', 'roundness']  # Add more if available (e.g., 'color_intensity', 'source')
    target_col = 'polymer_type'  # Multi-class: polymer type (includes 'non_plastic' for non-microplastics)
    
    # Check and map columns if needed
    available_features = [col for col in feature_cols if col in df.columns]
    if len(available_features) < len(feature_cols):
        logger.warning(f"Missing features: {set(feature_cols) - set(available_features)}. Using available: {available_features}")
        feature_cols = available_features
    
    if target_col not in df.columns:
        # Derive from 'is_microplastic' if present
        if 'is_microplastic' in df.columns:
            df[target_col] = np.where(df['is_microplastic'] == 1, np.random.choice(CLASSES[1:], len(df)), CLASSES[0])
        else:
            logger.error(f"Target column '{target_col}' not found. Cannot proceed.")
            raise ValueError("Target column missing.")
    
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Preprocessing: Impute missing, scale
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save preprocessors for inference
    import joblib
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features. Classes: {le.classes_}")
    return X, y_encoded, le, feature_cols

# Step 2: Train Tabular ML Model
def train_model(X, y):
    """
    Trains a RandomForest classifier with GridSearch for high accuracy.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
    # Save model
    import joblib
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Model saved.")
    return best_model

# Step 3: Particle Detection and Feature Extraction from Image
def detect_and_extract_features(image_path, pixel_to_micron=PIXEL_TO_MICRON, min_area=MIN_AREA_PIXELS):
    """
    Detects particles using OpenCV, extracts features (size in microns, aspect ratio, roundness).
    Returns: list of particles with features.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Input image {image_path} not found.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better detection in varying lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    particles = []
    for contour in contours:
        area_pixels = cv2.contourArea(contour)
        if area_pixels > min_area:
            # Equivalent diameter (microns): 2*sqrt(area/pi) in pixels, scaled by microns-per-pixel
            equiv_diameter_px = 2 * np.sqrt(area_pixels / np.pi)
            size_microns = equiv_diameter_px * pixel_to_micron
            
            # Bounding rect for aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 1.0
            
            # Roundness: 4*pi*area / (perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            roundness = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0.0
            
            particles.append({
                'bbox': (x, y, w, h),
                'size_microns': size_microns,
                'aspect_ratio': aspect_ratio,
                'roundness': roundness
            })
    
    logger.info(f"Detected {len(particles)} particles.")
    return particles

# Step 4: Classify Using Trained Model
def classify_particles(model, le, scaler, imputer, feature_cols, particles):
    """
    Extracts features for each particle and predicts polymer type.
    """
    if len(particles) == 0:
        return []
    
    # Prepare feature matrix
    features = []
    for p in particles:
        feat = [p.get(col, 0) for col in feature_cols]  # Default 0 if missing
        features.append(feat)

    # Ensure transformers receive named columns to avoid warnings
    features_arr = np.array(features)
    features_df = pd.DataFrame(features_arr, columns=feature_cols)
    features_imputed = imputer.transform(features_df)  # Impute if needed
    features_scaled = scaler.transform(features_imputed)

    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    classifications = []
    for i, pred in enumerate(predictions):
        class_name = le.inverse_transform([pred])[0]
        confidence = np.max(probabilities[i])
        is_micro = class_name != 'non_plastic'
        classifications.append({
            'polymer_type': class_name,
            'confidence': float(confidence),
            'is_microplastic': is_micro
        })
    
    return classifications

# Step 5: Main Pipeline
def main():
    """
    Full pipeline: Load/train model from dataset, process image, output results.
    
    """
    # model.py ke end me add karo

def analyze_image(image_path, pixel_to_micron=None, min_area=None):
    import joblib
    # Resolve calibration/min area defaults
    if pixel_to_micron is None:
        pixel_to_micron = PIXEL_TO_MICRON
    if min_area is None:
        min_area = MIN_AREA_PIXELS

    # 1. Load model + preprocessors (train if missing)
    try:
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_cols = ['size_microns', 'aspect_ratio', 'roundness']
    except Exception:
        logger.info("Model or preprocessors not found. Training a new model...")
        X, y, le, feature_cols = load_and_preprocess_dataset(CSV_PATH)
        model = train_model(X, y)
        # Reload saved preprocessors for consistency
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')

    # 2. Detect particles
    particles = detect_and_extract_features(image_path, pixel_to_micron=pixel_to_micron, min_area=min_area)

    if len(particles) == 0:
        return {"particles": [], "summary": {"total": 0, "microplastics": 0}}, image_path

    # 3. Classify particles
    classifications = classify_particles(model, le, scaler, imputer, feature_cols, particles)

    # 4. Build response
    microplastics_count = sum(1 for c in classifications if c['is_microplastic'])
    result = {
        "particles": [
            {
                "bbox": p["bbox"],
                "size_microns": p["size_microns"],
                "polymer_type": c["polymer_type"],
                "confidence": c["confidence"],
                "is_microplastic": c["is_microplastic"]
            }
            for p, c in zip(particles, classifications)
        ],
        "summary": {
            "total": len(particles),
            "microplastics": microplastics_count
        }
    }

    # Optional: save visualization (unique temp file per request)
    fd, output_image_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    for i, particle in enumerate(particles):
        x, y, w, h = particle['bbox']
        color = (0, 255, 0) if classifications[i]['is_microplastic'] else (255, 0, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        label = f"{classifications[i]['polymer_type']} ({classifications[i]['confidence']:.2f})"
        plt.text(x, y-5, label, color='white', fontsize=7, bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

    return result, output_image_path

    # Load dataset and train
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        logger.info("Loaded existing model.")
    except FileNotFoundError:
        logger.info("Loading dataset and training new model...")
        X, y, le, feature_cols = load_and_preprocess_dataset(CSV_PATH)
        model = train_model(X, y)
        # Preprocessors already saved in load_and_preprocess_dataset
    
    # Process image
    try:
        particles = detect_and_extract_features(INPUT_IMAGE_PATH)
        if len(particles) == 0:
            print("No particles detected.")
            return
        
        classifications = classify_particles(model, le, scaler, imputer, feature_cols, particles)
        
        # Output Results
        print("\n=== Detection and Classification Results ===")
        print(f"Total Number of Particles Detected: {len(particles)}")
        
        microplastics_count = sum(1 for c in classifications if c['is_microplastic'])
        polymer_breakdown = {cls: 0 for cls in CLASSES}
        for cls in classifications:
            polymer_breakdown[cls['polymer_type']] += 1
        
        for i, (particle, cls) in enumerate(zip(particles, classifications)):
            print(f"\nParticle {i+1}:")
            print(f"  Bounding Box: {particle['bbox']}")
            print(f"  Size (microns, area): {particle['size_microns']:.2f}")
            print(f"  Polymer Type: {cls['polymer_type']} (Confidence: {cls['confidence']:.4f})")
            print(f"  Is Microplastic: {'Yes' if cls['is_microplastic'] else 'No'}")
        
        print(f"\nSummary:")
        print(f"Total Microplastics: {microplastics_count}")
        print("Polymer Breakdown:")
        for poly, count in polymer_breakdown.items():
            print(f"  {poly}: {count}")
        
        # Visualize
        image = cv2.imread(INPUT_IMAGE_PATH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        for i, particle in enumerate(particles):
            x, y, w, h = particle['bbox']
            color = (0, 255, 0) if classifications[i]['is_microplastic'] else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            label = f"{classifications[i]['polymer_type'][:3]} ({classifications[i]['confidence']:.2f})"
            plt.text(x, y-10, label, color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
        plt.axis('off')
        plt.savefig('detection_output.png', bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")

if __name__ == "__main__":
    main()
