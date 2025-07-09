#!/usr/bin/env python3
"""
Script de diagnóstico para el reconocimiento facial
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle

# Importar la clase del archivo principal
from OmnifaceApp import FaceRecognitionSimulator

def test_recognition_detailed():
    """Prueba detallada del reconocimiento facial"""
    print("🔍 Diagnóstico detallado del reconocimiento facial")
    print("=" * 60)
    
    # Cargar datos
    try:
        with open('face_encodings.pkl', 'rb') as f:
            encodings = pickle.load(f)
        db = pd.read_csv('database.csv')
        
        print(f"✅ Encodings cargados: {list(encodings.keys())}")
        print(f"✅ Base de datos cargada: {len(db)} personas")
        
        for person_id, encoding in encodings.items():
            print(f"   - {person_id}: encoding shape = {encoding.shape}")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False
    
    # Crear simulador
    face_rec = FaceRecognitionSimulator()
    
    # Probar con la imagen existente
    image_path = "images/211436425.jpg"
    if os.path.exists(image_path):
        print(f"\n🖼️ Probando con imagen existente: {image_path}")
        
        try:
            # Cargar y procesar la imagen
            image = Image.open(image_path)
            image_array = np.array(image)
            
            print(f"   - Tamaño de imagen: {image_array.shape}")
            
            # Detectar rostros
            face_locations = face_rec.face_locations(image_array)
            print(f"   - Rostros detectados: {len(face_locations)}")
            
            if len(face_locations) > 0:
                print(f"   - Ubicaciones: {face_locations}")
                
                # Generar encoding
                face_encodings_new = face_rec.face_encodings(image_array, face_locations)
                print(f"   - Encodings generados: {len(face_encodings_new)}")
                
                if len(face_encodings_new) > 0:
                    new_encoding = face_encodings_new[0]
                    print(f"   - Nuevo encoding shape: {new_encoding.shape}")
                    
                    # Comparar con encoding existente
                    stored_encoding = encodings['211436425']
                    print(f"   - Encoding almacenado shape: {stored_encoding.shape}")
                    
                    # Comparar directamente
                    if len(stored_encoding) == len(new_encoding):
                        distance = np.linalg.norm(stored_encoding - new_encoding)
                        print(f"   - Distancia euclidiana: {distance:.4f}")
                        
                        # Probar comparación
                        matches = face_rec.compare_faces([stored_encoding], new_encoding, tolerance=0.6)
                        print(f"   - Match con tolerancia 0.6: {matches[0]}")
                        
                        # Probar con diferentes tolerancias
                        for tol in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
                            matches_tol = face_rec.compare_faces([stored_encoding], new_encoding, tolerance=tol)
                            print(f"   - Match con tolerancia {tol}: {matches_tol[0]}")
                    else:
                        print(f"   ⚠️ Tamaños de encoding diferentes!")
                        
            else:
                print("   ❌ No se detectaron rostros en la imagen")
                
        except Exception as e:
            print(f"   ❌ Error procesando imagen: {e}")
    
    # Probar con imagen nueva (simular nueva foto)
    print(f"\n📷 Probando con imagen aleatoria (simulando nueva foto)")
    try:
        # Crear imagen aleatoria con un rostro simulado
        test_image = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        
        # Agregar un patrón que pueda ser detectado como rostro
        cv2.rectangle(test_image, (100, 100), (200, 200), (150, 150, 150), -1)
        cv2.circle(test_image, (130, 130), 10, (0, 0, 0), -1)  # Ojo izquierdo
        cv2.circle(test_image, (170, 130), 10, (0, 0, 0), -1)  # Ojo derecho
        cv2.rectangle(test_image, (140, 160), (160, 180), (0, 0, 0), -1)  # Nariz/boca
        
        face_locations = face_rec.face_locations(test_image)
        print(f"   - Rostros detectados en imagen de prueba: {len(face_locations)}")
        
        if len(face_locations) > 0:
            encodings_test = face_rec.face_encodings(test_image, face_locations)
            if len(encodings_test) > 0:
                test_encoding = encodings_test[0]
                stored_encoding = encodings['211436425']
                
                distance = np.linalg.norm(stored_encoding - test_encoding)
                print(f"   - Distancia con encoding real: {distance:.4f}")
                
                matches = face_rec.compare_faces([stored_encoding], test_encoding, tolerance=0.6)
                print(f"   - Match: {matches[0]}")
                
    except Exception as e:
        print(f"   ❌ Error con imagen de prueba: {e}")
    
    print("\n💡 Recomendaciones para mejorar el reconocimiento:")
    print("   1. Verificar que la imagen tenga buena iluminación")
    print("   2. Asegurar que el rostro esté centrado y sin obstrucciones")
    print("   3. Considerar ajustar la tolerancia en la aplicación")
    print("   4. Verificar que el rostro sea detectado correctamente por OpenCV")
    
    return True

if __name__ == "__main__":
    test_recognition_detailed()
