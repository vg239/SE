# Versatile Verification Cloud Framework Demo

A comprehensive demonstration of Privacy-Preserving Machine Learning on Homomorphic Encryption and Zero-Knowledge Proofs for Cloud Computing, with a focus on Aadhaar identity verification.

## Media 
### This depicts the initial part of the generation of the MIBFHE keys
<img src='/media/client_side_1.png'></img>

### This represents the formation of the zk proofs here
<img src='/media/client_side_2.png'></img>

### This shows the ML risk score being generated from the existing model trained and response from the cloud server
<img src='/media/client_side_3.png'></img>

## Overview

This implementation demonstrates:

1. **Homomorphic Encryption (HE)**: Perform ML inference on encrypted data without decryption
2. **Zero-Knowledge Proofs (ZKP)**: Verify identity without revealing sensitive information
3. **Privacy-Preserving ML**: Anomaly detection on encrypted Aadhaar authentication data
4. **Cloud Computing**: Containerized services demonstrating secure cloud computation
5. **Benchmarking**: Performance analysis of cryptographic operations


## Project Structure

```
implementation/
├── frontend/
│   ├── frontend.py          # Streamlit frontend UI
│   └── requirements.txt      # Frontend dependencies
├── backend/
│   ├── cloud_server.py      # Flask cloud server with HE & ZKP
│   └── requirements.txt     # Backend dependencies
├── ml/
│   ├── train_model.py       # ML model training script
│   ├── ml_model_config.json # Trained model configuration
│   └── requirements.txt     # ML dependencies
├── benchmarks/
│   ├── benchmark_service.py # Benchmarking API service
│   └── requirements.txt     # Benchmark dependencies
├── admin/
│   ├── admin_panel.py        # Admin panel with logs & analytics
│   └── requirements.txt     # Admin dependencies
├── shared/
│   └── mibfhe_lib.py        # Core HE and ZKP libraries (shared)
├── Dockerfile.frontend      # Frontend container
├── Dockerfile.cloud         # Cloud server container
├── Dockerfile.benchmark     # Benchmark service container
├── Dockerfile.admin         # Admin panel container
├── docker-compose.yml       # Multi-container orchestration
├── Makefile                # Build automation
├── test_api.py             # API testing script
├── valid_users.txt         # Valid users database (name:uid:pincode)
├── QUICKSTART.md           # Quick start guide
├── README_VALID_USERS.md   # Valid users documentation
└── README.md               # This file
```

## Features

### 1. Aadhaar Data Structure
- Name, UID (12 digits), Pincode (6 digits)
- Optional: Date of Birth, Address
- All data encrypted before transmission

### 2. Zero-Knowledge Proofs
- Identity commitment generation
- Proof generation and verification
- Privacy-preserving authentication

### 3. Homomorphic Encryption
- MIBFHE (Multi-Identity Bounded FHE) scheme
- Encrypted addition and scalar multiplication
- ML inference on encrypted data

### 4. Privacy-Preserving ML
- Anomaly detection model (Logistic Regression)
- **Where ML is Used:**
  - ML model runs inference on **encrypted data** in the cloud server
  - All computations (multiplication, addition) happen on encrypted values
  - Cloud performs: `Enc(score) = w1×Enc(location) + w2×Enc(time) + w3×Enc(device) + w4×Enc(pincode) + bias`
  - Only the client can decrypt the final anomaly score
  - Cloud never sees plaintext risk scores or identity data
- Risk scoring on encrypted features:
  - Location risk (0-10)
  - Time risk (0-10)
  - Device trust (0-10)
  - Pincode risk (0-10)
- **ML Processing Flow:**
  1. Client encrypts risk factors using Homomorphic Encryption
  2. Cloud receives encrypted values (cannot decrypt)
  3. Cloud applies ML model weights using homomorphic operations
  4. Cloud computes weighted sum on encrypted data
  5. Cloud returns encrypted result
  6. Client decrypts to get final anomaly score

### 5. Cloud Computing Demo
- Containerized services
- Network isolation
- Health checks
- Audit logging

## Usage Example

1. **Open the frontend** at http://localhost:8501

2. **Enter Aadhaar data (default: Vatsal, UID: 123456789012, Pincode: 575025):**
   - Name: Vatsal (or any valid user from `valid_users.txt`)
   - UID: 123456789012
   - Pincode: 575025

3. **Set risk factors:**
   - Location Risk: 2 (low)
   - Time Risk: 1 (normal hours)
   - Device Trust: 8 (trusted device)
   - Pincode Risk: 1 (registered pincode)

4. **Click "Authenticate & Analyze Risk"**

5. **Observe:**
   - ZK Proof generation (hashing process shown step-by-step)
   - Data encryption (plaintext → encrypted blobs)
   - Cloud computation (ML inference on encrypted data)
   - Decrypted results (only client can decrypt)
   - Privacy guarantees (backend never sees plaintext)

6. **Test Invalid User:**
   - Try a name/UID/pincode combination NOT in `valid_users.txt`
   - Observe: ZK Proof verification fails
   - This demonstrates that only pre-registered hashes work

## Benchmarking

Access benchmark data via the benchmark service:

```bash
# Get latency benchmarks
curl http://localhost:5001/benchmarks/latency

# Get depth benchmarks
curl http://localhost:5001/benchmarks/depth

# Get threading benchmarks
curl http://localhost:5001/benchmarks/threading

# Get summary
curl http://localhost:5001/benchmarks/summary
```

## API Endpoints

### Cloud Server (Port 5000)

- `GET /health` - Health check
- `POST /secure_inference` - Secure ML inference on encrypted data
- `POST /verify_identity` - ZK identity verification
- `GET /audit` - Audit log
- `GET /logs` - Detailed operation logs (for admin panel)
- `GET /logs/stats` - Operation statistics

### Benchmark Service (Port 5001)

- `GET /health` - Health check
- `GET /benchmarks/latency` - Latency benchmarks
- `GET /benchmarks/depth` - Depth benchmarks
- `GET /benchmarks/threading` - Threading benchmarks
- `GET /benchmarks/summary` - Summary of all benchmarks

### Privacy Guarantees

- **Backend Never Sees Plaintext:**
  - Only SHA-256 hashes (commitments) are stored in the database
  - Only encrypted blobs are processed for ML inference
  - Backend cannot decrypt or recover original data
  - Even if backend is compromised, attacker gets only hashes and encrypted values

- **Adding New Users:**
  - Edit `valid_users.txt` file: `name:uid:pincode` (one per line)
  - Restart backend to load new users
  - Only the hash (commitment) is stored, never plaintext

- **ML Model Usage:**
  - ML model runs entirely on encrypted data
  - Homomorphic operations: multiplication and addition on ciphertexts
  - Model weights are public (applied to encrypted values)
  - Final score remains encrypted until client decryption





