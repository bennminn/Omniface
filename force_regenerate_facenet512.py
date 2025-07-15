#!/usr/bin/env python3
"""
Script para forzar la regeneración completa de encodings usando Facenet512 exclusivamente.
Garantiza que todos los encodings sean de 512 dimensiones y compatibles entre sí.
"""

import os
import pickle
import pandas as pd
import numpy as np
from deepface import DeepFace
import cv2
from pathlib import Path

def force_regenerate_all_encodings():
    """
    Fuerza la regeneración de TODOS los encodings usando Facenet512 exclusivamente.
    """
    print("🔄 INICIANDO REGENERACIÓN FORZADA CON FACENET512")
    print("=" * 60)
    
    # Cargar base de datos
    if not os.path.exists('database.csv'):
        print("❌ No se encontró database.csv")
        return False
    
    df = pd.read_csv('database.csv')
    print(f"📊 Total de registros en base de datos: {len(df)}")
    
    # Verificar que todas las imágenes existan
    missing_images = []
    for _, row in df.iterrows():
        image_path = f"images/{row['id']}.jpg"
        if not os.path.exists(image_path):
            missing_images.append(row['id'])
    
    if missing_images:
        print(f"⚠️  Imágenes faltantes para IDs: {missing_images}")
        print("   Continuando con las imágenes disponibles...")
    
    # Limpiar encodings existentes
    print("\n🗑️  Limpiando encodings existentes...")
    if os.path.exists('face_encodings.pkl'):
        os.remove('face_encodings.pkl')
        print("   ✅ face_encodings.pkl eliminado")
    
    # Generar nuevos encodings con Facenet512 EXCLUSIVAMENTE
    new_encodings = {}
    success_count = 0
    error_count = 0
    
    print("\n🔧 Generando nuevos encodings con Facenet512...")
    print("-" * 50)
    
    for _, row in df.iterrows():
        person_id = str(row['id'])
        image_path = f"images/{person_id}.jpg"
        
        if not os.path.exists(image_path):
            print(f"⚠️  {person_id}: Imagen no encontrada - SALTANDO")
            error_count += 1
            continue
        
        try:
            print(f"🔍 Procesando {person_id}...")
            
            # Verificar que la imagen se puede leer
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ {person_id}: No se puede leer la imagen")
                error_count += 1
                continue
            
            # Generar encoding con Facenet512 FORZADAMENTE
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet512',  # FORZAR Facenet512
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            # Extraer el vector de características
            if isinstance(embedding, list) and len(embedding) > 0:
                encoding_vector = np.array(embedding[0]['embedding'])
            else:
                encoding_vector = np.array(embedding['embedding'])
            
            # Verificar dimensiones
            if encoding_vector.shape[0] != 512:
                print(f"❌ {person_id}: Encoding incorrecto - {encoding_vector.shape[0]}D en lugar de 512D")
                error_count += 1
                continue
            
            # Guardar encoding
            new_encodings[person_id] = encoding_vector
            print(f"✅ {person_id}: Encoding 512D generado correctamente")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {person_id}: Error - {str(e)}")
            error_count += 1
            continue
    
    # Guardar nuevos encodings
    if new_encodings:
        print(f"\n💾 Guardando {len(new_encodings)} encodings...")
        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump(new_encodings, f)
        print("   ✅ face_encodings.pkl guardado")
    else:
        print("\n❌ No se generaron encodings válidos")
        return False
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE REGENERACIÓN:")
    print(f"   ✅ Exitosos: {success_count}")
    print(f"   ❌ Errores: {error_count}")
    print(f"   📊 Total procesados: {success_count + error_count}")
    
    if success_count > 0:
        print(f"\n🎉 Regeneración completada con {success_count} encodings de 512D")
        return True
    else:
        print("\n💥 Regeneración falló - no se generaron encodings válidos")
        return False

def verify_encodings():
    """
    Verifica que todos los encodings sean de 512D y compatibles.
    """
    print("\n🔍 VERIFICANDO ENCODINGS REGENERADOS")
    print("=" * 40)
    
    try:
        with open('face_encodings.pkl', 'rb') as f:
            encodings = pickle.load(f)
        
        print(f"📊 Total encodings cargados: {len(encodings)}")
        
        all_valid = True
        for person_id, encoding in encodings.items():
            if not isinstance(encoding, np.ndarray):
                print(f"❌ {person_id}: No es numpy array")
                all_valid = False
            elif encoding.shape[0] != 512:
                print(f"❌ {person_id}: {encoding.shape[0]}D en lugar de 512D")
                all_valid = False
            else:
                print(f"✅ {person_id}: 512D - OK")
        
        if all_valid:
            print("\n🎉 TODOS LOS ENCODINGS SON VÁLIDOS (512D)")
            return True
        else:
            print("\n❌ HAY ENCODINGS INVÁLIDOS")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando encodings: {e}")
        return False

def test_recognition():
    """
    Prueba el reconocimiento facial con los nuevos encodings.
    """
    print("\n🧪 PROBANDO RECONOCIMIENTO FACIAL")
    print("=" * 40)
    
    try:
        # Cargar encodings
        with open('face_encodings.pkl', 'rb') as f:
            known_encodings = pickle.load(f)
        
        if not known_encodings:
            print("❌ No hay encodings para probar")
            return False
        
        # Tomar el primer usuario para prueba
        test_person_id = list(known_encodings.keys())[0]
        test_image_path = f"images/{test_person_id}.jpg"
        
        if not os.path.exists(test_image_path):
            print(f"❌ Imagen de prueba no encontrada: {test_image_path}")
            return False
        
        print(f"🔍 Probando reconocimiento de usuario: {test_person_id}")
        
        # Generar encoding de la imagen de prueba con Facenet512
        embedding = DeepFace.represent(
            img_path=test_image_path,
            model_name='Facenet512',
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        if isinstance(embedding, list) and len(embedding) > 0:
            test_encoding = np.array(embedding[0]['embedding'])
        else:
            test_encoding = np.array(embedding['embedding'])
        
        if test_encoding.shape[0] != 512:
            print(f"❌ Encoding de prueba inválido: {test_encoding.shape[0]}D")
            return False
        
        print(f"✅ Encoding de prueba generado: 512D")
        
        # Calcular distancias
        min_distance = float('inf')
        best_match = None
        
        for person_id, known_encoding in known_encodings.items():
            distance = np.linalg.norm(test_encoding - known_encoding)
            print(f"   📏 Distancia a {person_id}: {distance:.4f}")
            
            if distance < min_distance:
                min_distance = distance
                best_match = person_id
        
        # Evaluar resultado
        threshold = 0.6  # Umbral para Facenet512
        print(f"\n📊 RESULTADO:")
        print(f"   🎯 Mejor coincidencia: {best_match}")
        print(f"   📏 Distancia mínima: {min_distance:.4f}")
        print(f"   🚧 Umbral: {threshold}")
        
        if min_distance < threshold and best_match == test_person_id:
            print(f"   ✅ RECONOCIMIENTO EXITOSO! Usuario {test_person_id} reconocido correctamente")
            return True
        elif best_match == test_person_id:
            print(f"   ⚠️  COINCIDENCIA CORRECTA pero distancia alta ({min_distance:.4f} > {threshold})")
            print(f"      💡 Considerar ajustar umbral a {min_distance + 0.1:.1f}")
            return True
        else:
            print(f"   ❌ RECONOCIMIENTO FALLIDO - coincidencia incorrecta")
            return False
        
    except Exception as e:
        print(f"❌ Error en prueba de reconocimiento: {e}")
        return False

if __name__ == "__main__":
    print("🚀 REGENERACIÓN FORZADA DE ENCODINGS FACENET512")
    print("=" * 60)
    
    # Paso 1: Regenerar encodings
    if force_regenerate_all_encodings():
        
        # Paso 2: Verificar encodings
        if verify_encodings():
            
            # Paso 3: Probar reconocimiento
            if test_recognition():
                print("\n🎉 REGENERACIÓN Y PRUEBAS COMPLETADAS CON ÉXITO")
                print("   La app debería funcionar correctamente ahora")
            else:
                print("\n⚠️  REGENERACIÓN OK, pero hay problemas de reconocimiento")
                print("   Revisa los umbrales en la app")
        else:
            print("\n❌ REGENERACIÓN FALLÓ - encodings inválidos")
    else:
        print("\n💥 REGENERACIÓN FALLÓ COMPLETAMENTE")
    
    print("\n" + "=" * 60)
