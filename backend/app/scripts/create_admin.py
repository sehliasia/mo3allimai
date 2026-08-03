import getpass
from app.database.session import SessionLocal
from app.models.user import UserRole
from app.repositories.user_repository import UserRepository
from app.core.security import hash_password

def main():
    name = input("Full name: ").strip()
    email = input("Email: ").strip().lower()
    password = getpass.getpass("Password: ")
    if len(password) < 6: raise SystemExit("Password must contain at least 6 characters.")
    db = SessionLocal()
    try:
        users = UserRepository(db)
        if users.get_by_email(email): raise SystemExit("Email already exists.")
        users.create(full_name=" ".join(name.split()), email=email, password_hash=hash_password(password), role=UserRole.admin)
        print("Admin created.")
    finally: db.close()

if __name__ == "__main__": main()
