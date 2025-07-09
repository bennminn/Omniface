#!/usr/bin/env python3
"""
Script para regenerar todos los encodings faciales con tamaño consistente
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle

# Importar la clase del archivo principal
from OmnifaceApp import FaceRecognitionSimulator

def regenerate_encodings():
    """Regenerar todos los encodings con tamaño consistente"""
    print("🔄 Regenerando encodings faciales...")
    
    # Cargar base de datos
    try:
        db = pd.read_csv('database.csv')
        print(f"✅ Base de datos cargada: {len(db)} personas")
    except Exception as e:
        print(f"❌ Error cargando base de datos: {e}")
        return False
    
    # Crear nuevo diccionario de encodings
    new_encodings = {}
    face_rec = FaceRecognitionSimulator()
    
    for idx, row in db.iterrows():
        person_id = row['id']
        image_path = row['imagen_path']
        
        print(f"📷 Procesando {person_id}: {image_path}")
        
        if os.path.exists(image_path):
            try:
                # Cargar imagen
                image = Image.open(image_path)
                image_array = np.array(image)
                
                # Detectar rostros
                face_locations = face_rec.face_locations(image_array)
                print(f"   - Rostros detectados: {len(face_locations)}")
                
                if len(face_locations) > 0:
                    # Generar encodings
                    face_encodings = face_rec.face_encodings(image_array, face_locations)
                    
                    if len(face_encodings) > 0:
                        # Usar el primer rostro detectado
                        encoding = face_encodings[0]
                        new_encodings[person_id] = encoding
                        print(f"   ✅ Encoding generado: shape = {encoding.shape}")
                    else:
                        print(f"   ❌ No se pudo generar encoding")
                else:
                    print(f"   ❌ No se detectó rostro en la imagen")
                    
            except Exception as e:
                print(f"   ❌ Error procesando {person_id}: {e}")
        else:
            print(f"   ❌ Imagen no encontrada: {image_path}")
    
    # Guardar nuevos encodings
    if new_encodings:
        try:
            # Hacer backup del archivo anterior
            if os.path.exists('face_encodings.pkl'):
                os.rename('face_encodings.pkl', 'face_encodings_backup.pkl')
                print("📦 Backup del archivo anterior creado")
            
            # Guardar nuevos encodings
            with open('face_encodings.pkl', 'wb') as f:
                pickle.dump(new_encodings, f)
            
            print(f"✅ Nuevos encodings guardados: {len(new_encodings)} personas")
            
            # Verificar tamaños
            for person_id, encoding in new_encodings.items():
                print(f"   - {person_id}: {encoding.shape}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error guardando encodings: {e}")
            return False
    else:
        print("❌ No se generaron encodings")
        return False

if __name__ == "__main__":
    success = regenerate_encodings()
    if success:
        print("\n🎉 Encodings regenerados exitosamente!")
        print("💡 Ahora puedes probar el reconocimiento facial nuevamente")
    else:
        print("\n❌ Error regenerando encodings")
