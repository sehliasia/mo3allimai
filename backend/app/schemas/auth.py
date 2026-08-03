from pydantic import BaseModel, EmailStr, Field
from .user import UserRead
class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
class RegisterResponse(BaseModel):
    message: str
    user: UserRead
class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
