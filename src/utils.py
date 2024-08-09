import bcrypt
import firebase_admin
import base64
from firebase_admin import firestore


def encrypt_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return base64.b64encode(hashed_password).decode('utf-8')  # Codifica el hash en base64 y lo convierte a string

def verify_password(provided_password, stored_password):
    stored_password_bytes = base64.b64decode(stored_password.encode('utf-8'))
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password_bytes)


def is_email_unique(email, db):
    users_ref = db.collection('users')
    query = users_ref.where('email', '==', email).limit(1).get()
    return not query


def get_new_user_id():
    db = firestore.client()
    counter_ref = db.collection('counters').document('users')

    @firestore.transactional
    def update_counter(transaction):
        snapshot = counter_ref.get(transaction=transaction)
        new_user_id = snapshot.get('count') + 1
        transaction.update(counter_ref, {'count': new_user_id})
        return new_user_id

    transaction = db.transaction()
    new_user_id = update_counter(transaction)
    return new_user_id

def extract_years(message):
    years = [int(s) for s in message.split() if s.isdigit()]
    return years[0] if years else 0

def extract_technologies(message):
    return message.replace("and", ",").split(",")

def extract_level(message):
    if "junior" in message.lower():
        return "Junior"
    elif "intermediate" in message.lower():
        return "Intermediate"
    elif "senior" in message.lower():
        return "Senior"
    else:
        return "Unknown"
