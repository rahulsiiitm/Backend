from datetime import datetime
from config import db

def validate_user_id(user_id):
    if not user_id or str(user_id).strip() == '' or str(user_id) in ['null', 'undefined']:
        return False
    return True

def update_user_activity(user_id):
    try:
        user_ref = db.collection("users").document(user_id)
        user_ref.set({"lastActive": datetime.now()}, merge=True)
    except Exception:
        pass
