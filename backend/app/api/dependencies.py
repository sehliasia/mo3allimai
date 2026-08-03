import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.database.session import get_db
from app.models.user import User, UserRole
from app.repositories.user_repository import UserRepository
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    settings = get_settings()
    try: payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]); user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError): raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    user = UserRepository(db).get_by_id(user_id)
    if not user or not user.is_active: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive user")
    return user
def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.admin: raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user
