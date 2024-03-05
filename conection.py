import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Usa las credenciales para autenticar y conectar con Firebase
cred = credentials.Certificate('C:/Users/rctva/PycharmProjects/CodingCoach/codingcoach-firebase-adminsdk-32vz3-9f11a60448.json')
firebase_admin.initialize_app(cred)

# Obtén una referencia a la base de datos
db = firestore.client()

doc_ref = db.collection('users').document('1')
doc_ref.set({
    'name': 'Kim Namjoom',
    'email': 'rkive@example.com',
    'age': 29
})

doc_ref = db.collection('users').document('1')
try:
    doc = doc_ref.get()
    if doc.exists:
        print('Document data:', doc.to_dict())
    else:
        print('No such document!')
except Exception as e:
    print('Error:', e)