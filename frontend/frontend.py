import streamlit as st
import time
import requests
import json
import sys
import os
from datetime import datetime

# Add shared directory to path - support both Docker and local development
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_paths = [
    os.path.join(current_dir, '..', 'shared'),  # Local: frontend/../shared
    '/app/shared',                               # Docker: /app/shared
    os.path.join(os.getcwd(), 'shared'),         # Current dir/shared
]

for path in shared_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        sys.path.insert(0, abs_path)
        break

from mibfhe_lib import MIBFHEKeyGenerator, MIBFHEEncryption, ZKProofSystem, AadhaarData

# Cloud URL configuration
import os

# Check if running in Docker
def is_docker():
    return os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true"

# Determine cloud URL
if os.environ.get("CLOUD_URL"):
    CLOUD_URL = os.environ.get("CLOUD_URL")
elif is_docker():
    # In Docker, use service name for internal communication
    CLOUD_URL = "http://cloud-server:5000"
else:
    # Local development
    try:
        CLOUD_URL = st.secrets.get("CLOUD_URL", "http://localhost:5000")
    except:
        CLOUD_URL = "http://localhost:5000"

# st.set_page_config(
#     page_title="Secure Cloud V&V Demo", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

st.title("Versatile Verification Cloud Framework")
st.markdown("### Privacy-Preserving ML & ZKP for Cloud Computing (Aadhaar Case Study)")

# --- Sidebar: Aadhaar Data Input ---
st.sidebar.header("Aadhaar Identity Data")
st.sidebar.markdown("*All data is encrypted before transmission*")

name = st.sidebar.text_input("Full Name", "Vatsal")
uid = st.sidebar.text_input("Aadhaar UID (12 digits)", "123456789012", max_chars=12)
pincode = st.sidebar.text_input("Pincode (6 digits)", "575025", max_chars=6)
dob = st.sidebar.date_input("Date of Birth", value=None)
address = st.sidebar.text_area("Address (Optional)", "")

st.sidebar.markdown("---")
st.sidebar.header("Activity Context (Risk Factors)")
st.sidebar.markdown("*These are encrypted and sent to cloud for ML analysis*")

loc_risk = st.sidebar.slider("Location Risk Score", 0, 10, 2, 
                             help="0 = Home location, 10 = Foreign/unusual location")
time_risk = st.sidebar.slider("Time Risk Score", 0, 10, 1,
                             help="0 = Normal hours, 10 = Unusual hours (e.g., 3 AM)")
device_risk = st.sidebar.slider("Device Trust Score", 0, 10, 8,
                               help="0 = Unknown device, 10 = Trusted device")
pincode_risk = st.sidebar.slider("Pincode Risk Score", 0, 10, 1,
                                 help="0 = Registered pincode, 10 = Unusual pincode")

# --- Main Content Area ---
# Use single column for better vertical flow
st.subheader("1. Client-Side Processing")
st.markdown("**Privacy-Preserving Operations**")

if st.button("Authenticate & Analyze Risk", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Key Generation
    status_text.text("Step 1/5: Generating Homomorphic Encryption Keys...")
    progress_bar.progress(20)
    keygen = MIBFHEKeyGenerator()
    pk, sk = keygen.generate_keypair()
    encryptor = MIBFHEEncryption(pk)
    
    # Display MIBFHE Keys
    st.markdown("**MIBFHE Keys Generated**")
    with st.expander("View MIBFHE Keys", expanded=False):
        st.markdown("**Public Key (pk):**")
        st.code(str(pk)[:100] + "..." if len(str(pk)) > 100 else str(pk), language="text")
        st.markdown("**Secret Key (sk):**")
        st.code(str(sk)[:100] + "..." if len(str(sk)) > 100 else str(sk), language="text")
        st.info("Keys are used for homomorphic encryption. Public key is shared with cloud, secret key stays on client.")
    
    # Step 2: Create Aadhaar Data Object
    status_text.text("Step 2/5: Creating Aadhaar Data Structure...")
    progress_bar.progress(40)
    aadhaar_data = AadhaarData(
        name=name,
        uid=uid,
        pincode=pincode,
        dob=str(dob) if dob else None,
        address=address
    )
    
    # Step 3: Generate ZK Proof with detailed hashing visualization
    status_text.text("Step 3/5: Generating Zero-Knowledge Proof...")
    progress_bar.progress(60)
    
    # Show hashing process step-by-step
    st.markdown("#### Zero-Knowledge Proof Generation Process")
    
    # Step 3a: Show input data
    st.markdown("**Step 3a: Input Data**")
    st.code(f"""
Name:    {name}
UID:     {uid}
Pincode: {pincode}
    """, language="text")
    
    # Step 3b: Show concatenation
    st.markdown("**Step 3b: Data Concatenation**")
    concatenated_data = f"{name}:{uid}:{pincode}"
    st.code(concatenated_data, language="text")
    st.info("Data is concatenated with colons as separators")
    
    # Step 3c: Show hash generation
    st.markdown("**Step 3c: SHA-256 Hash Generation**")
    import hashlib
    hash_input = concatenated_data.encode()
    hash_obj = hashlib.sha256(hash_input)
    commitment = hash_obj.hexdigest()
    
    st.code(f"""
Input (bytes): {hash_input}
Algorithm:     SHA-256
Output (hex):  {commitment}
    """, language="text")
    
    st.info(f"Commitment Hash: `{commitment[:32]}...` (64 characters total)")
    
    # Step 3d: Generate full ZK proof
    zk_identity = ZKProofSystem.create_anonymized_identity(aadhaar_data)
    zk_proof = zk_identity['zk_proof']
    
    st.markdown("**Step 3d: ZK Proof Creation**")
    st.json(zk_proof)
    
    # Generate QR Code for ZK Proof
    st.markdown("---")
    st.markdown("#### QR Code for ZK Proof")
    
    try:
        import qrcode
        from PIL import Image
        import io
        
        qr_data = json.dumps({
            "commitment": commitment,
            "proof_hash": zk_proof.get('proof_hash', '')[:16],
            "timestamp": datetime.now().isoformat()
        })
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")

        img_buffer = io.BytesIO()
        qr_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        st.image(img_buffer, caption="ZK Proof QR Code", width=200)
    except ImportError:
        st.warning("QR code generation requires: pip install qrcode[pil]")
    except Exception as e:
        st.warning(f"QR code generation failed: {e}")
    
    with st.expander("View Complete ZK Identity Object", expanded=False):
        st.json(zk_identity)
        st.info("The cloud can verify this proof without learning your identity!")
    
    st.markdown(f"**ZK Proof Generated** - Commitment: `{commitment[:16]}...{commitment[-8:]}`")
    
    # Step 4: Encrypt Risk Factors
    status_text.text("Step 4/5: Encrypting Risk Factors...")
    progress_bar.progress(80)
    
    enc_loc = encryptor.encrypt(loc_risk)
    enc_time = encryptor.encrypt(time_risk)
    enc_dev = encryptor.encrypt(device_risk)
    enc_pincode = encryptor.encrypt(pincode_risk)
    
    with st.expander("View Encrypted Data", expanded=False):
        st.code(f"""
Encrypted Location Risk: {str(enc_loc)[:30]}...
Encrypted Time Risk: {str(enc_time)[:30]}...
Encrypted Device Risk: {str(enc_dev)[:30]}...
Encrypted Pincode Risk: {str(enc_pincode)[:30]}...
        """)
    
    st.markdown("**All Data Encrypted** - Cloud cannot see plaintext values")
    
    # Step 5: Send to Cloud
    status_text.text("Step 5/5: Sending to Cloud Server...")
    progress_bar.progress(90)
    
    payload = {
        "public_key": pk,
        "zk_commitment": commitment,
        "zk_proof": zk_proof,
        "enc_location": enc_loc,
        "enc_time": enc_time,
        "enc_device": enc_dev,
        "enc_pincode": enc_pincode
    }
    
    try:
        response = requests.post(
            f"{CLOUD_URL}/secure_inference", 
            json=payload,
            timeout=30
        )
        res_data = response.json()
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        if res_data['status'] == 'success':
            st.markdown("---")
            st.subheader("2. Cloud Response & Results")
            st.markdown("**Decrypted Results**")
            
            # Display verification status - normal text instead of green boxes
            verification = res_data.get('verification', {})
            if verification.get('zk_proof_verified'):
                st.markdown("✓ **Zero-Knowledge Proof Verified**")
            if verification.get('identity_verified'):
                st.markdown("✓ **Identity Verified** (without revealing data)")
            
            # Decrypt the anomaly score
            enc_score = res_data['encrypted_anomaly_score']
            dec_score = encryptor.decrypt(enc_score, sk)
            
            # Display ML Model Usage
            st.markdown("#### ML Model Processing")
            model_meta = res_data['model_metadata']
            st.markdown(f"""
            **Model Computation:**
            - All operations performed on encrypted data
            - Homomorphic multiplication: Enc(x) * weight
            - Homomorphic addition: Enc(x1) + Enc(x2)
            - Final score computed without decryption
            - Threshold: {model_meta.get('risk_threshold', 'N/A')}
            - Bias Adjustment: {model_meta.get('bias_adjustment', 'N/A')}
            """)
            
            # Display results
            st.metric(
                label="Anomaly Risk Score", 
                value=f"{dec_score:.2f}",
                delta=f"Threshold: {res_data['model_metadata']['risk_threshold']}"
            )
            
            threshold = res_data['model_metadata']['risk_threshold']
            
            # Show ML computation breakdown
            st.markdown("#### ML Model Computation Breakdown")
            st.markdown(f"""
            **Formula Applied (on encrypted data):**
            ```
            Enc(Score) = w1 × Enc(location) + w2 × Enc(time) + 
                             w3 × Enc(device) + w4 × Enc(pincode) + bias
            ```
            
            **Your Inputs (encrypted before sending):**
            - Location Risk: {loc_risk} → Encrypted
            - Time Risk: {time_risk} → Encrypted
            - Device Trust: {device_risk} → Encrypted
            - Pincode Risk: {pincode_risk} → Encrypted
            
            **Computation (in cloud, on encrypted data):**
            - All multiplications and additions performed homomorphically
            - Cloud never decrypted your values
            - Final result: {dec_score:.2f} (decrypted by you only)
            """)
            
            if dec_score > threshold:
                st.error("**ANOMALY DETECTED!**")
                st.warning("High risk score indicates suspicious activity pattern.")
                st.info("**Action**: Additional authentication required.")
            else:
                st.success("**ACCESS GRANTED**")
                st.info("Risk score within acceptable range.")
            
            # Display model metadata
            with st.expander("Model Metadata", expanded=False):
                st.json(res_data['model_metadata'])
            
            # Display raw response
            with st.expander("Raw Cloud Response", expanded=False):
                st.json(res_data)
            
            # Privacy demonstration
            st.info("""
            **Privacy Guarantees:**
            - Cloud never saw your plaintext identity or risk scores
            - All computation performed on encrypted data
            - ZK Proof verified identity without revealing it
            - Only you can decrypt the final result
            """)
            
        else:
            error_msg = res_data.get('msg', 'Unknown error')
            st.error(f"Cloud Error: {error_msg}")
            
            # Show explanation for verification failure
            if "Verification Failed" in error_msg or "verification failed" in error_msg.lower():
                st.warning("""
                **Verification Failed - Invalid User**
                
                This user's hash is not in the backend database. This demonstrates:
                - Only pre-registered users (in valid_users.txt) can verify
                - Backend checks hash against database
                - Invalid hashes are rejected
                - Backend never learns why verification failed (privacy preserved)
                
                **To add this user:**
                1. Add entry to valid_users.txt: name:uid:pincode
                2. Restart backend: docker-compose restart cloud-server
                3. Try again
                """)
            
    except requests.exceptions.ConnectionError:
        st.error("**Connection Failed**: Could not reach cloud server.")
        st.info(f"Make sure the cloud server is running at: `{CLOUD_URL}`")
        st.code(f"docker-compose up cloud-server", language="bash")
    except requests.exceptions.Timeout:
        st.error("**Timeout**: Cloud server took too long to respond.")
    except Exception as e:
        st.error(f"**Error**: {str(e)}")
        st.exception(e)

# --- Information Section ---
st.markdown("---")
st.subheader("How It Works")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    **Homomorphic Encryption**
    - Data encrypted before sending to cloud
    - Cloud performs ML inference on encrypted data
    - Only client can decrypt results
    """)

with col_info2:
    st.markdown("""
    **Zero-Knowledge Proofs**
    - Proves identity without revealing it
    - Cloud verifies you're authorized
    - No sensitive data exposed
    """)

with col_info3:
    st.markdown("""
    **Privacy-Preserving ML**
    - Anomaly detection on encrypted data
    - Risk scoring without data exposure
    - Proactive threat detection
    """)

# --- Footer ---
st.markdown("---")
st.caption("""
**Research Demo**: Versatile Verification of Security in Cloud Computing  
Framework using Machine Learning on Homomorphic Encryption and Zero Knowledge Proofs
""")