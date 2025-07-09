#!/usr/bin/env python3
"""
Script de prueba para verificar que OmniFace funciona correctamente
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

def test_opencv():
    """Probar que OpenCV funciona correctamente"""
    print("🔍 Probando OpenCV...")
    try:
        # Crear una imagen de prueba
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:50, :50] = [255, 0, 0]  # Cuadrado rojo
        
        # Verificar detector de rostros
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("❌ Error: No se pudo cargar el detector de rostros")
            return False
        
        print("✅ OpenCV funcionando correctamente")
        return True
    except Exception as e:
        print(f"❌ Error con OpenCV: {e}")
        return False

def test_face_recognition_simulator():
    """Probar el simulador de face_recognition"""
    print("🧠 Probando simulador de reconocimiento facial...")
    try:
        # Importar la clase del archivo principal
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from OmnifaceApp import FaceRecognitionSimulator
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Probar detección de rostros
        face_rec = FaceRecognitionSimulator()
        locations = face_rec.face_locations(test_image)
        print(f"✅ Detección de rostros: {len(locations)} rostros encontrados")
        
        # Probar encodings
        if locations:
            encodings = face_rec.face_encodings(test_image, locations)
            print(f"✅ Generación de encodings: {len(encodings)} encodings creados")
        
        print("✅ Simulador de reconocimiento facial funcionando")
        return True
    except Exception as e:
        print(f"❌ Error con el simulador: {e}")
        return False

def test_file_operations():
    """Probar operaciones de archivos"""
    print("📁 Probando operaciones de archivos...")
    try:
        # Crear directorio de prueba
        test_dir = "test_images"
        os.makedirs(test_dir, exist_ok=True)
        
        # Crear imagen de prueba
        test_image = Image.new('RGB', (100, 100), color='red')
        test_path = os.path.join(test_dir, "test.jpg")
        test_image.save(test_path)
        
        # Verificar que se guardó
        if os.path.exists(test_path):
            print("✅ Guardado de imágenes funcionando")
            
            # Limpiar
            os.remove(test_path)
            os.rmdir(test_dir)
            print("✅ Operaciones de archivos funcionando")
            return True
        else:
            print("❌ Error guardando imagen de prueba")
            return False
    except Exception as e:
        print(f"❌ Error con operaciones de archivos: {e}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🚀 Iniciando pruebas de OmniFace...")
    print("=" * 50)
    
    tests = [
        test_opencv,
        test_face_recognition_simulator,
        test_file_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! OmniFace está listo para usar.")
        print("🌐 Ejecuta 'streamlit run OmnifaceApp.py' para iniciar la aplicación")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
