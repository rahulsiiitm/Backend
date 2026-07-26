import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

# Initialize Firebase
firebase_key = os.getenv("FIREBASE_KEY")
if not firebase_key:
    print("Error: FIREBASE_KEY not found in environment variables.")
    exit(1)

cred = credentials.Certificate(json.loads(firebase_key))
firebase_admin.initialize_app(cred)
db = firestore.client()

def migrate_crops():
    print("Starting crops migration...")
    users_ref = db.collection("users")
    users = users_ref.stream()
    
    migrated_count = 0
    for user in users:
        crops_ref = users_ref.document(user.id).collection("crops")
        crops = crops_ref.stream()
        
        for crop in crops:
            crop_data = crop.to_dict()
            updates = {}
            
            # Standardize plantedDate vs sowedDate
            planted_date = crop_data.get("plantedDate") or crop_data.get("sowedDate") or crop_data.get("date") or datetime.now().strftime("%Y-%m-%d")
            
            # Enforce schema
            updates["name"] = str(crop_data.get("name", "Unknown Crop"))
            updates["type"] = str(crop_data.get("type", "Unknown Type"))
            updates["plantedDate"] = str(planted_date)
            updates["area"] = str(crop_data.get("area", "Unknown Area"))
            
            if "sowedDate" in crop_data:
                updates["sowedDate"] = firestore.DELETE_FIELD
                
            # Perform update
            if updates:
                crops_ref.document(crop.id).update(updates)
                migrated_count += 1
                
    print(f"Migrated {migrated_count} crops.")

def migrate_chats():
    print("Starting chats migration...")
    users_ref = db.collection("users")
    users = users_ref.stream()
    
    migrated_count = 0
    for user in users:
        chats_ref = users_ref.document(user.id).collection("chats")
        chats = chats_ref.stream()
        
        for chat in chats:
            chat_data = chat.to_dict()
            updates = {}
            
            if "createdAt" not in chat_data:
                updates["createdAt"] = datetime.now()
            if "updatedAt" not in chat_data:
                updates["updatedAt"] = datetime.now()
            if "messages" not in chat_data:
                updates["messages"] = []
            if "lastMessage" not in chat_data:
                updates["lastMessage"] = ""
                
            if updates:
                chats_ref.document(chat.id).update(updates)
                migrated_count += 1
                
    print(f"Migrated {migrated_count} chats.")

if __name__ == "__main__":
    print("Starting Database Migration...")
    migrate_crops()
    migrate_chats()
    print("Migration completed successfully!")
