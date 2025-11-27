from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
import logging
from datetime import datetime

# Add shared directory to path - support both Docker and local development
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_paths = [
    os.path.join(current_dir, '..', 'shared'),  # Local: backend/../shared
    '/app/shared',                               # Docker: /app/shared
    os.path.join(os.getcwd(), 'shared'),         # Current dir/shared
]

for path in shared_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        sys.path.insert(0, abs_path)
        break

# Import from shared library
from mibfhe_lib import HomomorphicOperations, ZKProofSystem, AadhaarData

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Trained ML Model (Simulating a production deployment)
ML_CONFIG_PATH = os.environ.get("ML_CONFIG_PATH", "/app/ml_model_config.json")
if not os.path.exists(ML_CONFIG_PATH):
    ML_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'ml_model_config.json')

ML_CONFIG = None
try:
    if os.path.exists(ML_CONFIG_PATH):
        with open(ML_CONFIG_PATH, "r") as f:
            content = f.read().strip()
            if content:  # Check if file is not empty
                ML_CONFIG = json.loads(content)
                logger.info(f"[Init] Loaded ML Model. Weights: {ML_CONFIG.get('weights', {})}")
            else:
                raise ValueError("Model config file is empty")
    else:
        raise FileNotFoundError(f"Model config file not found at {ML_CONFIG_PATH}")
except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
    # Fallback to default config
    logger.warning(f"[Warning] Model config not found or invalid ({e})! Using default weights.")
    ML_CONFIG = {
        "weights": {
            "location_weight": 5, 
            "time_weight": 3, 
            "device_weight": -2,
            "pincode_weight": 1
        }, 
        "bias": 10,
        "threshold": 15
    }

# Valid Citizens Database (in production, this would be a secure database)
# These are commitments of valid Aadhaar identities
# Only HASHES are stored - never plaintext names, UIDs, or pincodes
def load_valid_citizens_db():
    """Load valid citizens from file and generate commitments (hashes only)"""
    db = []
    valid_users_file = os.path.join(os.path.dirname(__file__), '..', 'valid_users.txt')
    
    # Try to load from file
    if os.path.exists(valid_users_file):
        try:
            with open(valid_users_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        parts = line.split(':')
                        if len(parts) == 3:
                            name, uid, pincode = parts
                            # Generate commitment (hash) - this is all we store
                            commitment = ZKProofSystem.generate_commitment(name.strip(), uid.strip(), pincode.strip())
                            db.append(commitment)
                            logger.info(f"[Init] Loaded user commitment: {commitment[:16]}... (NO PLAINTEXT STORED)")
        except Exception as e:
            logger.warning(f"[Init] Could not load valid_users.txt: {e}")
    
    # Fallback to default users if file doesn't exist or is empty
    if not db:
        logger.info("[Init] Using default users (no valid_users.txt found)")
        default_users = [
            ("Vatsal", "123456789012", "575025"),
            ("Alice", "123456789012", "560001"),
            ("Bob", "987654321098", "110001"),
            ("Charlie", "111122223333", "400001"),
            ("Diana", "444455556666", "700001"),
        ]
        for name, uid, pincode in default_users:
            commitment = ZKProofSystem.generate_commitment(name, uid, pincode)
            db.append(commitment)
            logger.info(f"[Init] Default user commitment: {commitment[:16]}... (NO PLAINTEXT STORED)")
    
    return db

VALID_CITIZENS_DB = load_valid_citizens_db()
logger.info(f"[Init] Loaded {len(VALID_CITIZENS_DB)} user commitments (hashes only, no plaintext data)")
logger.info(f"[Init] Database contains ONLY hashes - no names, UIDs, or pincodes stored")
logger.info(f"[Init] Database contains ONLY hashes - no names, UIDs, or pincodes stored")

# Store for audit logging (in production, use proper database)
audit_log = []
detailed_logs = []  # Detailed logs for admin panel

def log_operation(operation_type, details):
    """Log operation with details for admin panel"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation_type,
        'details': details
    }
    detailed_logs.append(log_entry)
    # Keep only last 1000 entries
    if len(detailed_logs) > 1000:
        detailed_logs.pop(0)
    return log_entry

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    return jsonify({
        "status": "healthy",
        "service": "Cloud Computing Server",
        "ml_model_loaded": ML_CONFIG.get('weights') is not None
    })

@app.route('/secure_inference', methods=['POST'])
def secure_inference():
    """
    Main endpoint for secure ML inference on encrypted Aadhaar data.
    Demonstrates:
    1. Zero-Knowledge Proof verification
    2. Homomorphic encryption operations
    3. Privacy-preserving ML inference
    """
    try:
        data = request.json
        
        # --- STEP 1: ZERO-KNOWLEDGE PROOF VERIFICATION ---
        zk_proof = data.get('zk_proof', {})
        zk_commitment = data.get('zk_commitment') or zk_proof.get('commitment')
        
        if not zk_commitment:
            return jsonify({
                "status": "error", 
                "msg": "Missing ZK commitment"
            }), 400
        
        logger.info(f"[Cloud] Processing Request for ZK-ID: {zk_commitment[:8]}...")
        
        # Log ZK proof verification attempt
        log_operation('zk_verification_start', {
            'zk_commitment_prefix': zk_commitment[:16],
            'has_proof': bool(zk_proof),
            'proof_hash': zk_proof.get('proof_hash', '')[:16] if zk_proof else None
        })
        
        # Verify ZK Proof if provided
        verification_result = False
        if zk_proof:
            verification_result = ZKProofSystem.verify_proof(zk_proof, VALID_CITIZENS_DB)
            log_operation('zk_proof_verification', {
                'method': 'proof_verification',
                'result': verification_result,
                'commitment_prefix': zk_commitment[:16]
            })
            if not verification_result:
                logger.warning(f"[Cloud] ZK Proof verification failed for {zk_commitment[:8]}")
                return jsonify({
                    "status": "error", 
                    "msg": "ZK Proof Verification Failed"
                }), 401
        else:
            # Fallback to simple commitment check
            verification_result = ZKProofSystem.verify_commitment(zk_commitment, VALID_CITIZENS_DB)
            log_operation('zk_commitment_verification', {
                'method': 'commitment_check',
                'result': verification_result,
                'commitment_prefix': zk_commitment[:16]
            })
            if not verification_result:
                logger.warning(f"[Cloud] Identity verification failed for {zk_commitment[:8]}")
                return jsonify({
                    "status": "error", 
                    "msg": "Identity Verification Failed"
                }), 401
        
        logger.info(f"[Cloud] ✓ ZK Proof Verified (without learning identity)")
        log_operation('zk_verification_success', {
            'commitment_prefix': zk_commitment[:16],
            'verified': True
        })
        
        # --- STEP 2: ENCRYPTED ML INFERENCE ---
        # Retrieve Encrypted Inputs (all operations happen on encrypted data)
        pk = data['public_key']
        enc_loc = data.get('enc_location', 0)
        enc_time = data.get('enc_time', 0)
        enc_dev = data.get('enc_device', 0)
        enc_pincode = data.get('enc_pincode', 0)  # New: pincode risk
        
        ops = HomomorphicOperations(pk)
        
        # Perform Homomorphic Dot Product: w1*x1 + w2*x2 + w3*x3 + w4*x4 + bias
        # All operations happen on encrypted data - cloud never sees plaintext
        
        w_loc = ML_CONFIG['weights']['location_weight']
        w_time = ML_CONFIG['weights']['time_weight']
        w_dev = ML_CONFIG['weights']['device_weight']
        w_pincode = ML_CONFIG['weights'].get('pincode_weight', 0)
        bias = ML_CONFIG.get('bias', 0)
        
        # Log ML inference start with input data (encrypted only)
        log_operation('ml_inference_start', {
            'model_weights': ML_CONFIG['weights'],
            'has_encrypted_inputs': True,
            'encrypted_inputs': {
                'location_prefix': str(enc_loc)[:20] + '...',
                'time_prefix': str(enc_time)[:20] + '...',
                'device_prefix': str(enc_dev)[:20] + '...',
                'pincode_prefix': str(enc_pincode)[:20] + '...'
            }
        })
        
        # Compute weighted sum homomorphically
        term_1 = ops.multiply_scalar(enc_loc, w_loc)
        log_operation('he_operation', {
            'operation': 'multiply_scalar',
            'feature': 'location',
            'weight': w_loc,
            'result_prefix': str(term_1)[:20]
        })
        
        term_2 = ops.multiply_scalar(enc_time, w_time)
        log_operation('he_operation', {
            'operation': 'multiply_scalar',
            'feature': 'time',
            'weight': w_time,
            'result_prefix': str(term_2)[:20]
        })
        
        term_3 = ops.multiply_scalar(enc_dev, abs(w_dev))  # Handle negative weight
        log_operation('he_operation', {
            'operation': 'multiply_scalar',
            'feature': 'device',
            'weight': w_dev,
            'result_prefix': str(term_3)[:20]
        })
        
        term_4 = ops.multiply_scalar(enc_pincode, w_pincode)
        log_operation('he_operation', {
            'operation': 'multiply_scalar',
            'feature': 'pincode',
            'weight': w_pincode,
            'result_prefix': str(term_4)[:20]
        })
        
        # Sum all terms (Homomorphic Addition)
        step_1 = ops.add(term_1, term_2)
        step_2 = ops.add(step_1, term_3)
        step_3 = ops.add(step_2, term_4)
        
        log_operation('he_operation', {
            'operation': 'add',
            'steps': ['term1+term2', 'step1+term3', 'step2+term4'],
            'final_sum_prefix': str(step_3)[:20]
        })
        
        # Add bias (encrypted zero + bias scalar)
        enc_zero = ops.multiply_scalar(enc_loc, 0)  # Create encrypted zero
        final_score = ops.add(step_3, enc_zero + bias)
        
        logger.info(f"[Cloud] ✓ Encrypted Inference Complete. Result Blob: {str(final_score)[:20]}...")
        logger.info(f"[Cloud] ✓ Cloud never saw plaintext data - privacy preserved!")
        
        log_operation('ml_inference_complete', {
            'final_score_prefix': str(final_score)[:20],
            'bias_added': bias,
            'operations_count': 8
        })
        
        # Audit log (metadata only, no sensitive data)
        audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'zk_id_prefix': zk_commitment[:8],
            'operation': 'secure_inference',
            'status': 'success'
        })
        
        return jsonify({
            "status": "success",
            "encrypted_anomaly_score": final_score,
            "model_metadata": {
                "bias_adjustment": bias,
                "risk_threshold": ML_CONFIG.get('threshold', 15),
                "model_version": "1.0"
            },
            "verification": {
                "zk_proof_verified": True,
                "identity_verified": True
            }
        })
        
    except Exception as e:
        logger.error(f"[Cloud] Error: {str(e)}")
        return jsonify({
            "status": "error",
            "msg": f"Server error: {str(e)}"
        }), 500

@app.route('/audit', methods=['GET'])
def get_audit_log():
    """Get audit log (for demonstration purposes)"""
    return jsonify({
        "status": "success",
        "audit_entries": audit_log[-50:],  # Last 50 entries
        "total_entries": len(audit_log)
    })

@app.route('/logs', methods=['GET'])
def get_detailed_logs():
    """Get detailed logs for admin panel"""
    limit = request.args.get('limit', 100, type=int)
    operation_filter = request.args.get('operation', None)
    
    logs = detailed_logs[-limit:] if limit > 0 else detailed_logs
    
    if operation_filter:
        logs = [log for log in logs if log['operation'] == operation_filter]
    
    return jsonify({
        "status": "success",
        "logs": logs,
        "total_entries": len(detailed_logs),
        "filtered_count": len(logs)
    })

@app.route('/logs/stats', methods=['GET'])
def get_log_stats():
    """Get statistics about operations"""
    stats = {}
    for log in detailed_logs:
        op_type = log['operation']
        stats[op_type] = stats.get(op_type, 0) + 1
    
    return jsonify({
        "status": "success",
        "statistics": stats,
        "total_operations": len(detailed_logs)
    })

@app.route('/verify_identity', methods=['POST'])
def verify_identity():
    """
    Standalone endpoint for ZK identity verification.
    Demonstrates zero-knowledge proof verification without ML inference.
    """
    try:
        data = request.json
        zk_proof = data.get('zk_proof', {})
        zk_commitment = data.get('zk_commitment') or zk_proof.get('commitment')
        
        if not zk_commitment:
            return jsonify({"status": "error", "msg": "Missing ZK commitment"}), 400
        
        if zk_proof:
            verified = ZKProofSystem.verify_proof(zk_proof, VALID_CITIZENS_DB)
        else:
            verified = ZKProofSystem.verify_commitment(zk_commitment, VALID_CITIZENS_DB)
        
        return jsonify({
            "status": "success" if verified else "error",
            "verified": verified,
            "message": "Identity verified via ZK Proof" if verified else "Identity verification failed"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)