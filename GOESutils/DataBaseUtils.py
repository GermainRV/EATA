from firebase_admin import credentials, storage, initialize_app
import os

ruta_credenciales = 'GOESutils/key.json'
cred = credentials.Certificate(ruta_credenciales)
initialize_app(cred, {'storageBucket': 'eata-project.appspot.com'})

def subir(full_local_path, destine_path, filename):
    bucket = storage.bucket()
    destine_path = destine_path.replace("\\", "/").replace(" ","_")
    filename = filename.replace(" ","_")
    full_destine_path = f"{destine_path}/{filename}"  
    # full_destine_path = os.path.join(destine_path, filename)
    blob = bucket.blob(full_destine_path)
    blob.upload_from_filename(full_local_path)
    print(f"File {os.path.basename(full_local_path)} uploaded to '{destine_path}'")

def eliminar(carpeta):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=carpeta)
    for blob in blobs:
        if blob.name.endswith('/') or blob.name == carpeta:
            continue
        blob.delete()
        print(f"Imagen eliminada: {blob.name}")




