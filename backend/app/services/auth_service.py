from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.core.security import hash_password, verify_password, create_access_token
from app.models.user import User, UserRole
from app.repositories.user_repository import UserRepository
class AuthService:
    def __init__(self, db: Session): self.users = UserRepository(db)
    def register(self, full_name: str, email: str, password: str) -> User:
        normalized_email = email.strip().lower(); cleaned_name = " ".join(full_name.split())
        if self.users.get_by_email(normalized_email): raise HTTPException(status_code=409, detail="Email already registered")
        return self.users.create(full_name=cleaned_name, email=normalized_email, password_hash=hash_password(password), role=UserRole.teacher)
    def authenticate(self, email: str, password: str) -> User:
        user = self.users.get_by_email(email.strip().lower())
        if not user or user.is_deleted or not verify_password(password, user.password_hash): raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password", headers={"WWW-Authenticate": "Bearer"})
        if not user.is_active: raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")
        return user
    def login(self, email: str, password: str) -> tuple[str, User]:
        user = self.authenticate(email, password)
        return create_access_token(str(user.id), user.role.value), user
