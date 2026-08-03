from datetime import datetime, timezone
from math import ceil
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from app.api.dependencies import require_admin
from app.database.session import get_db
from app.models.user import User, UserRole
from app.schemas.user import UserRead

router = APIRouter(prefix="/admin", tags=["admin"])
def teacher_or_404(db: Session, teacher_id: int) -> User:
    teacher = db.get(User, teacher_id)
    if not teacher or teacher.role != UserRole.teacher or teacher.is_deleted: raise HTTPException(404, "Teacher not found")
    return teacher
@router.get("/teachers")
def teachers(search: str | None = None, is_active: bool | None = None, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), sort_order: str = Query("desc", pattern="^(asc|desc)$"), _: User = Depends(require_admin), db: Session = Depends(get_db)):
    filters = [User.role == UserRole.teacher, User.is_deleted.is_(False)]
    if search: filters.append(or_(User.full_name.ilike(f"%{search.strip()}%"), User.email.ilike(f"%{search.strip()}%")))
    if is_active is not None: filters.append(User.is_active == is_active)
    total = db.scalar(select(func.count()).select_from(User).where(*filters)) or 0
    order = User.created_at.asc() if sort_order == "asc" else User.created_at.desc()
    items = db.scalars(select(User).where(*filters).order_by(order).offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [UserRead.model_validate(item).model_dump() for item in items], "page": page, "page_size": page_size, "total": total, "total_pages": ceil(total / page_size) if total else 0}
@router.get("/teachers/{teacher_id}", response_model=UserRead)
def teacher(teacher_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)): return teacher_or_404(db, teacher_id)
@router.patch("/teachers/{teacher_id}/status", response_model=UserRead)
def status(teacher_id: int, data: dict, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    teacher = teacher_or_404(db, teacher_id)
    if teacher.id == admin.id: raise HTTPException(403, "Cannot modify own account")
    if "is_active" not in data or not isinstance(data["is_active"], bool): raise HTTPException(422, "is_active is required")
    teacher.is_active = data["is_active"]; db.commit(); db.refresh(teacher); return teacher
@router.delete("/teachers/{teacher_id}", response_model=UserRead)
def delete_teacher(teacher_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    teacher = teacher_or_404(db, teacher_id)
    if teacher.id == admin.id: raise HTTPException(403, "Cannot delete own account")
    teacher.is_active = False; teacher.is_deleted = True; teacher.deleted_at = datetime.now(timezone.utc); teacher.deleted_by = admin.id; db.commit(); db.refresh(teacher); return teacher
@router.get("/statistics")
def statistics(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    base = [User.role == UserRole.teacher, User.is_deleted.is_(False)]
    total = db.scalar(select(func.count()).select_from(User).where(*base)) or 0
    active = db.scalar(select(func.count()).select_from(User).where(*base, User.is_active.is_(True))) or 0
    month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    new = db.scalar(select(func.count()).select_from(User).where(*base, User.created_at >= month)) or 0
    return {"total_teachers": total, "active_teachers": active, "inactive_teachers": total - active, "new_teachers_this_month": new}
