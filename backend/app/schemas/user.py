from datetime import datetime
from pydantic import BaseModel, ConfigDict
from app.models.user import UserRole
class UserRead(BaseModel):
    id: int
    full_name: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True)
