import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import joblib
import os

def generate_synthetic_aadhaar_data(n_samples=10000):
    """
    Generates a synthetic dataset of Aadhaar authentication attempts.
    Features:
    - Location Risk (0-10): Distance from registered location/unusual IP.
    - Time Risk (0-10): Login hour weirdness (e.g., 3 AM).
    - Device Trust (0-10): 10 is known device, 0 is new/untrusted.
    - Pincode Risk (0-10): 0 = registered pincode, 10 = unusual pincode.
    
    Target:
    - 1 (Anomaly/Attack), 0 (Normal)
    """
    np.random.seed(42)
    
    # Generate Normal User Behavior (Low Risk) - 95% of data
    n_normal = int(n_samples * 0.95)
    normal_loc = np.random.normal(1, 1, n_normal)  # Mostly safe locations
    normal_time = np.random.normal(2, 2, n_normal)  # Normal hours
    normal_dev = np.random.normal(9, 1, n_normal)   # Trusted devices
    normal_pincode = np.random.normal(1, 1, n_normal)  # Registered pincode
    
    # Generate Attacker Behavior (High Risk) - 5% of data
    n_attack = n_samples - n_normal
    attack_loc = np.random.normal(8, 1, n_attack)      # Foreign/unusual IPs
    attack_time = np.random.normal(8, 2, n_attack)      # Weird hours
    attack_dev = np.random.normal(2, 2, n_attack)       # New/untrusted devices
    attack_pincode = np.random.normal(8, 1, n_attack)   # Unusual pincodes
    
    # Combine
    loc = np.concatenate([normal_loc, attack_loc])
    time = np.concatenate([normal_time, attack_time])
    dev = np.concatenate([normal_dev, attack_dev])
    pincode = np.concatenate([normal_pincode, attack_pincode])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_attack)])
    
    # Clip values to 0-10 range for easier integer HE mapping later
    data = pd.DataFrame({
        'location_risk': np.clip(loc, 0, 10).astype(int),
        'time_risk': np.clip(time, 0, 10).astype(int),
        'device_trust': np.clip(dev, 0, 10).astype(int),
        'pincode_risk': np.clip(pincode, 0, 10).astype(int),
        'is_anomaly': y.astype(int)
    })
    
    return data

def semantic_check_model(model, X_test, y_test):
    """
    Performs semantic checks on the trained model to ensure it makes sense.
    Checks:
    1. High risk features should increase anomaly score
    2. Low risk features should decrease anomaly score
    3. Model weights should be interpretable
    """
    print("\n[Semantic Check] Validating Model Logic...")
    
    # Get model weights
    weights = model.coef_[0]
    feature_names = ['location_risk', 'time_risk', 'device_trust', 'pincode_risk']
    
    # Check 1: Location and Time risk should have positive weights (increase risk)
    location_weight = weights[0]
    time_weight = weights[1]
    
    if location_weight > 0:
        print(f"    ✓ Location risk weight is positive ({location_weight:.4f}) - increases anomaly score")
    else:
        print(f"    ⚠ Location risk weight is negative ({location_weight:.4f}) - unexpected")
    
    if time_weight > 0:
        print(f"    ✓ Time risk weight is positive ({time_weight:.4f}) - increases anomaly score")
    else:
        print(f"    ⚠ Time risk weight is negative ({time_weight:.4f}) - unexpected")
    
    # Check 2: Device trust should have negative weight (decreases risk)
    device_weight = weights[2]
    if device_weight < 0:
        print(f"    ✓ Device trust weight is negative ({device_weight:.4f}) - decreases anomaly score (correct)")
    else:
        print(f"    ⚠ Device trust weight is positive ({device_weight:.4f}) - unexpected")
    
    # Check 3: Test with high-risk vs low-risk scenarios
    print("\n    Testing Semantic Scenarios:")
    
    # High risk scenario
    high_risk_sample = pd.DataFrame({
        'location_risk': [9],
        'time_risk': [9],
        'device_trust': [1],
        'pincode_risk': [9]
    })
    high_risk_score = model.predict_proba(high_risk_sample)[0][1]
    print(f"    High Risk Scenario (loc=9, time=9, dev=1, pin=9): Anomaly Probability = {high_risk_score:.4f}")
    
    # Low risk scenario
    low_risk_sample = pd.DataFrame({
        'location_risk': [1],
        'time_risk': [1],
        'device_trust': [9],
        'pincode_risk': [1]
    })
    low_risk_score = model.predict_proba(low_risk_sample)[0][1]
    print(f"    Low Risk Scenario (loc=1, time=1, dev=9, pin=1): Anomaly Probability = {low_risk_score:.4f}")
    
    # Verify high risk > low risk
    if high_risk_score > low_risk_score:
        print(f"    ✓ Semantic check passed: High risk ({high_risk_score:.4f}) > Low risk ({low_risk_score:.4f})")
    else:
        print(f"    ⚠ Semantic check failed: High risk ({high_risk_score:.4f}) <= Low risk ({low_risk_score:.4f})")
    
    # Check 4: Feature importance
    print("\n    Feature Importance (absolute weights):")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'weight': weights,
        'abs_weight': np.abs(weights)
    }).sort_values('abs_weight', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"      {row['feature']}: {row['weight']:.4f} (importance: {row['abs_weight']:.4f})")
    
    # Convert numpy types to Python native types for JSON serialization
    weights_dict = {}
    for name, weight in zip(feature_names, weights):
        weights_dict[name] = float(weight)  # Convert numpy float to Python float
    
    return {
        'weights': weights_dict,
        'high_risk_score': float(high_risk_score),
        'low_risk_score': float(low_risk_score),
        'semantic_check_passed': bool(high_risk_score > low_risk_score)
    }

def train_and_export():
    """
    Trains a logistic regression model on synthetic Aadhaar authentication data.
    Exports model weights in a format compatible with homomorphic encryption.
    """
    print("=" * 60)
    print("Training Privacy-Preserving ML Model for Anomaly Detection")
    print("=" * 60)
    
    print("\n[1/4] Generating Synthetic Aadhaar Dataset...")
    df = generate_synthetic_aadhaar_data(n_samples=10000)
    print(f"    ✓ Created {len(df)} samples")
    print(f"    ✓ Anomaly Rate: {df['is_anomaly'].mean():.2%}")
    print(f"    ✓ Features: location_risk, time_risk, device_trust, pincode_risk")
    
    # Train Model
    print("\n[2/4] Training Logistic Regression Model...")
    X = df[['location_risk', 'time_risk', 'device_trust', 'pincode_risk']]
    y = df['is_anomaly']
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # We use Logistic Regression because it's a linear model (Dot Product)
    # This maps perfectly to Homomorphic Encryption (Weighted Sum).
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[3/4] Evaluating Model Performance...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"    ✓ Training Accuracy: {train_acc:.4f}")
    print(f"    ✓ Test Accuracy: {test_acc:.4f}")
    
    print("\n    Classification Report:")
    print(classification_report(y_test, test_preds, target_names=['Normal', 'Anomaly']))
    
    # Extract Weights (Coefficients)
    # The Cloud needs these to compute: w1*x1 + w2*x2 + w3*x3 + w4*x4 + bias
    weights = model.coef_[0]
    bias = model.intercept_[0]
    
    print(f"\n    Model Weights:")
    print(f"      Location:  {weights[0]:.4f}")
    print(f"      Time:      {weights[1]:.4f}")
    print(f"      Device:    {weights[2]:.4f}")
    print(f"      Pincode:   {weights[3]:.4f}")
    print(f"      Bias:      {bias:.4f}")
    
    # Perform semantic checks
    semantic_results = semantic_check_model(model, X_test, y_test)
    
    # Scale weights to integers for our specific MIBFHE library (which likes ints)
    # e.g., if weight is 0.5, we scale by 10 to get 5.
    SCALE_FACTOR = 10
    
    export_data = {
        "model_type": "logistic_regression",
        "scale_factor": SCALE_FACTOR,
        "weights": {
            "location_weight": int(weights[0] * SCALE_FACTOR),
            "time_weight": int(weights[1] * SCALE_FACTOR),
            "device_weight": int(weights[2] * SCALE_FACTOR),  # Usually negative (trust reduces risk)
            "pincode_weight": int(weights[3] * SCALE_FACTOR)
        },
        "bias": int(bias * SCALE_FACTOR),
        "threshold": 50,  # Threshold for anomaly detection in scaled domain
        "metrics": {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc)
        },
        "semantic_checks": semantic_results
    }
    
    print("\n[4/4] Exporting Model Configuration...")
    print(f"    ✓ Scaled Weights: {export_data['weights']}")
    
    # Save to multiple locations for Docker compatibility
    output_paths = [
        os.path.join(os.path.dirname(__file__), "ml_model_config.json"),
        "/app/ml_model_config.json"  # Docker path
    ]
    
    for path in output_paths:
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(path) if os.path.dirname(path) else '.'
            if parent_dir and parent_dir != '.':
                os.makedirs(parent_dir, exist_ok=True)
            
            # Remove if it's a directory (shouldn't happen, but safety check)
            if os.path.exists(path) and os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                print(f"    ⚠ Removed directory at '{path}', creating file instead")
            
            # Write JSON file
            with open(path, "w") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            print(f"    ✓ Saved to '{path}'")
        except PermissionError:
            print(f"    ⚠ Permission denied for '{path}' (skipping)")
        except Exception as e:
            print(f"    ⚠ Could not save to '{path}': {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    print("\nThe model is ready for homomorphic encryption inference.")
    print("Cloud server can use these weights to perform encrypted ML inference.")

if __name__ == "__main__":
    train_and_export()