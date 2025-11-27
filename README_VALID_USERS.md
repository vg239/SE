# Valid Users Database

## Overview

The backend maintains a database of valid users for ZK proof verification. **CRITICALLY**, this database contains **ONLY HASHES** (SHA-256 commitments), never plaintext names, UIDs, or pincodes.

## File Format

Edit `valid_users.txt` to add or remove users:

```
# Format: name:uid:pincode
# One entry per line
# Lines starting with # are comments

Vatsal:123456789012:575025
Alice:123456789012:560001
Bob:987654321098:110001
```

## How It Works

1. **File Format:** `name:uid:pincode` (colon-separated)
2. **Backend Processing:**
   - Reads each line from `valid_users.txt`
   - Generates SHA-256 hash: `SHA256(name:uid:pincode)`
   - Stores ONLY the hash in `VALID_CITIZENS_DB`
   - **Never stores plaintext data**

3. **Verification:**
   - Client sends ZK commitment (hash) in request
   - Backend checks if hash exists in database
   - If hash matches → Verified
   - If hash doesn't match → Verification Failed
   - Backend never learns the actual name/UID/pincode

## Example

### Adding a User

1. Edit `valid_users.txt`:
   ```
   NewUser:999988887777:123456
   ```

2. Restart backend:
   ```bash
   docker-compose restart cloud-server
   ```

3. Backend will:
   - Read the file
   - Generate hash: `SHA256("NewUser:999988887777:123456")`
   - Store hash in database
   - Log: `Loaded user commitment: abc123... (NO PLAINTEXT STORED)`

### Testing Invalid User

Try using a name/UID/pincode combination NOT in `valid_users.txt`:
- Frontend: Name="InvalidUser", UID="000000000000", Pincode="999999"
- Result: ZK Proof Verification Failed
- Reason: Hash doesn't exist in database

## Security Properties

1. **Privacy:** Backend never stores or sees plaintext identity data
2. **Verification:** Only pre-registered hashes can verify successfully
3. **One-way:** Hashes cannot be reversed to recover original data
4. **Collision-resistant:** Different inputs produce different hashes

## Current Valid Users

See `valid_users.txt` for the current list. Default users:
- Vatsal (UID: 123456789012, Pincode: 575025)
- Alice (UID: 123456789012, Pincode: 560001)
- Bob (UID: 987654321098, Pincode: 110001)
- Charlie (UID: 111122223333, Pincode: 400001)
- Diana (UID: 444455556666, Pincode: 700001)

## Verification Flow

```
Client                          Backend
  |                                |
  |-- Generate hash from data ---->|
  |   SHA256("Vatsal:123...:575025")|
  |                                |
  |-- Send hash in request ------->|
  |   commitment: "261f2511..."    |
  |                                |
  |                                |-- Check hash in database
  |                                |-- Hash found? → Verified
  |                                |-- Hash not found? → Failed
  |<-- Verification result --------|
  |   verified: true/false         |
```

**Important:** Backend only sees the hash, never "Vatsal", "123456789012", or "575025".

