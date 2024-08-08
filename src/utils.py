import bcrypt


def encrypt_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password


def verify_password(input_password, stored_password):
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_password)


def is_email_unique(email, db):
    users_ref = db.collection('users')
    query = users_ref.where('email', '==', email).limit(1).get()
    return not query  # Retorna True si el email es único, False si ya está en uso
