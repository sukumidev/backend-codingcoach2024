import bcrypt
import firebase_admin
import base64
from firebase_admin import firestore

def encrypt_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return base64.b64encode(hashed_password).decode('utf-8')  # Codifica el hash en base64 y lo convierte a string


def verify_password(input_password, stored_password):
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_password)


def is_email_unique(email, db):
    users_ref = db.collection('users')
    query = users_ref.where('email', '==', email).limit(1).get()
    return not query  # Retorna True si el email es único, False si ya está en uso




def get_new_user_id():
    db = firestore.client()
    counter_ref = db.collection('counters').document('users')

    # Incrementa el contador de forma atómica
    @firestore.transactional
    def update_counter(transaction):
        snapshot = counter_ref.get(transaction=transaction)
        new_user_id = snapshot.get('count') + 1
        transaction.update(counter_ref, {'count': new_user_id})
        return new_user_id

    transaction = db.transaction()
    new_user_id = update_counter(transaction)
    return new_user_id
