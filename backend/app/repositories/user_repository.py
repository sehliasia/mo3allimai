from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.user import User, UserRole
class UserRepository:
    def __init__(self, db: Session): self.db = db
    def get_by_email(self, email: str) -> User | None: return self.db.scalar(select(User).where(User.email == email))
    def get_by_id(self, user_id: int) -> User | None: return self.db.get(User, user_id)
    def create(self, *, full_name: str, email: str, password_hash: str, role: UserRole = UserRole.teacher) -> User:
        user = User(full_name=full_name, email=email, password_hash=password_hash, role=role)
        self.db.add(user); self.db.commit(); self.db.refresh(user)
        return user
