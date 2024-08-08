import bcrypt

def encrypt_password(password):
    # Genera un salt
    salt = bcrypt.gensalt()
    # Encripta la contraseña
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

# Uso de la función
password = "mi_contraseña_secreta"
hashed_password = encrypt_password(password)
print("Contraseña encriptada:", hashed_password)

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

# Verificar la contraseña
is_correct = check_password("mi_contraseña", hashed_password)
print("¿La contraseña es correcta?", is_correct)
