"""
Authentication Routes - Signup, Login, User Info
Uses JSON file for simple user storage
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, EmailStr
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime

router = APIRouter()

# Data file path
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"


def load_users() -> dict:
    """Load users from JSON file"""
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {"users": []}


def save_users(data: dict):
    """Save users to JSON file"""
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def hash_password(password: str) -> str:
    """Hash password with SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token() -> str:
    """Generate a random auth token"""
    return secrets.token_hex(32)


# Request/Response models
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


@router.post("/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """Register a new user"""
    data = load_users()
    
    # Check if email already exists
    for user in data["users"]:
        if user["email"] == request.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    token = generate_token()
    new_user = {
        "id": len(data["users"]) + 1,
        "name": request.name,
        "email": request.email,
        "password_hash": hash_password(request.password),
        "token": token,
        "created_at": datetime.now().isoformat()
    }
    
    data["users"].append(new_user)
    save_users(data)
    
    return AuthResponse(
        token=token,
        user={"id": new_user["id"], "name": new_user["name"], "email": new_user["email"]}
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login existing user"""
    data = load_users()
    password_hash = hash_password(request.password)
    
    for user in data["users"]:
        if user["email"] == request.email and user["password_hash"] == password_hash:
            # Generate new token on login
            new_token = generate_token()
            user["token"] = new_token
            save_users(data)
            
            return AuthResponse(
                token=new_token,
                user={"id": user["id"], "name": user["name"], "email": user["email"]}
            )
    
    raise HTTPException(status_code=401, detail="Invalid email or password")


@router.get("/me")
async def get_current_user(authorization: str = Header(None)):
    """Get current user from token in header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(" ")[1]
    data = load_users()
    
    for user in data["users"]:
        stored_token = user.get("token")
        if stored_token == token:
            return {"id": user["id"], "name": user["name"], "email": user["email"]}
    
    raise HTTPException(status_code=401, detail="Invalid token")
