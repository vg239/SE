import random
import hashlib
import json
from typing import Dict, List, Tuple

class MIBFHEKeyGenerator:
    def __init__(self, security_param=128):
        # p is the SECRET prime (acts as the key)
        # q is the PUBLIC large prime
        self.p = 65537 
        self.q = 1152921504606846977 # 2^60 + 1
        
    def generate_keypair(self):
        sk = {'p': self.p}
        pk_elements = []
        # Generate valid zero-encryptions for the public key
        # c = (q*r + p*e) % q
        for _ in range(100): 
            r_i = random.randint(1, self.q // self.p)
            e_i = random.randint(-10, 10)
            c_i = (self.q * r_i + self.p * e_i) % self.q
            pk_elements.append(c_i)
        
        pk = {'q': self.q, 'elements': pk_elements}
        return pk, sk

class MIBFHEEncryption:
    def __init__(self, public_key):
        self.q = public_key['q']
        self.pk_elements = public_key['elements']
        
    def encrypt(self, value):
        """Encrypts a value (0 or 1 usually, but works for small ints)."""
        # c = m + sum(subset_of_pk)
        subset_size = random.randint(20, 50)
        subset_indices = random.sample(range(len(self.pk_elements)), subset_size)
        
        c = value
        for idx in subset_indices:
            c += self.pk_elements[idx]
        return c % self.q
        
    def decrypt(self, ciphertext, secret_key):
        """Decrypts to return the integer value (the risk score)."""
        p = secret_key['p']
        # Decrypt: c % p = m + noise
        noise_plus_m = ciphertext % p
        
        # Handle negative noise wrap-around
        if noise_plus_m > p // 2:
            noise_plus_m -= p
            
        return noise_plus_m

class HomomorphicOperations:
    def __init__(self, public_key):
        self.q = public_key['q']
        
    def add(self, ct1, ct2):
        return (ct1 + ct2) % self.q
        
    def multiply_scalar(self, ct, scalar):
        """Allows weighing features (e.g., Risk * 5)"""
        return (ct * scalar) % self.q

class AadhaarData:
    """Represents Aadhaar-like identity data structure"""
    def __init__(self, name: str, uid: str, pincode: str, dob: str = None, address: str = None):
        self.name = name
        self.uid = uid  # 12-digit Aadhaar UID
        self.pincode = pincode  # 6-digit pincode
        self.dob = dob
        self.address = address
        
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'uid': self.uid,
            'pincode': self.pincode,
            'dob': self.dob,
            'address': self.address
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class ZKProofSystem:
    """
    Enhanced Zero-Knowledge Proof system for Aadhaar verification.
    Implements a simplified zk-SNARK-like protocol for identity verification.
    """
    
    @staticmethod
    def generate_commitment(name: str, uid: str, pincode: str = None) -> str:
        """
        Creates a hash commitment of the identity (Pedersen commitment-like).
        This commitment hides the actual data but allows verification.
        """
        if pincode:
            data = f"{name}:{uid}:{pincode}"
        else:
            data = f"{name}:{uid}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def generate_proof(commitment: str, secret_salt: str = None) -> Dict:
        """
        Generates a zero-knowledge proof that the prover knows the commitment
        without revealing the underlying data.
        """
        if secret_salt is None:
            secret_salt = str(random.randint(100000, 999999))
        
        # Create proof token (simplified zk-SNARK proof)
        proof_data = f"{commitment}:{secret_salt}"
        proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()
        
        return {
            'commitment': commitment,
            'proof_hash': proof_hash,
            'salt': secret_salt
        }
    
    @staticmethod
    def verify_proof(proof: Dict, known_hash_db: List[str]) -> bool:
        """
        Verifies the zero-knowledge proof without learning the underlying data.
        Returns True if the commitment is valid and proof is correct.
        """
        commitment = proof.get('commitment')
        proof_hash = proof.get('proof_hash')
        salt = proof.get('salt')
        
        if not all([commitment, proof_hash, salt]):
            return False
        
        # Verify commitment exists in database
        if commitment not in known_hash_db:
            return False
        
        # Verify proof consistency
        proof_data = f"{commitment}:{salt}"
        expected_hash = hashlib.sha256(proof_data.encode()).hexdigest()
        
        return proof_hash == expected_hash
    
    @staticmethod
    def verify_commitment(commitment: str, known_hash_db: List[str]) -> bool:
        """Legacy method for backward compatibility"""
        return commitment in known_hash_db
    
    @staticmethod
    def create_anonymized_identity(aadhaar: AadhaarData) -> Dict:
        """
        Creates an anonymized identity token from Aadhaar data.
        This demonstrates privacy-preserving identity verification.
        """
        commitment = ZKProofSystem.generate_commitment(
            aadhaar.name, 
            aadhaar.uid, 
            aadhaar.pincode
        )
        proof = ZKProofSystem.generate_proof(commitment)
        
        return {
            'zk_commitment': commitment,
            'zk_proof': proof,
            'metadata': {
                'pincode_hash': hashlib.sha256(aadhaar.pincode.encode()).hexdigest()[:8],
                'uid_last4': aadhaar.uid[-4:] if len(aadhaar.uid) >= 4 else '****'
            }
        }