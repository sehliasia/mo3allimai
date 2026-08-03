from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.api.dependencies import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.core.security import create_access_token
from app.schemas.auth import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, TokenResponse
from app.schemas.user import UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    user = AuthService(db).register(data.full_name, str(data.email), data.password)
    return {"message": "Account created successfully", "user": user}

@router.post("/login", response_model=LoginResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    token, user = AuthService(db).login(str(data.email), data.password)
    return {"access_token": token, "token_type": "bearer", "user": user}

@router.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = AuthService(db).authenticate(form_data.username.strip().lower(), form_data.password)
    return {"access_token": create_access_token(str(user.id), user.role.value), "token_type": "bearer"}

@router.get("/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)):
    return user
