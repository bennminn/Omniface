#!/usr/bin/env python3
"""
Script de prueba para verificar que face_recognition funciona correctamente
después de la migración del simulador.
"""

import sys

def test_face_recognition_import():
    """Prueba que se puede importar face_recognition."""
    try:
        import face_recognition
        print("✅ face_recognition importado correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando face_recognition: {e}")
        print("💡 Instala con: pip install face_recognition")
        return False

def test_basic_functionality():
    """Prueba funcionalidad básica de face_recognition."""
    try:
        import face_recognition
        import numpy as np
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Probar detección de rostros
        face_locations = face_recognition.face_locations(test_image)
        print(f"✅ face_locations funciona (encontrados: {len(face_locations)} rostros)")
        
        # Probar encodings (aunque no haya rostros)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)
        print(f"✅ face_encodings funciona (encodings: {len(face_encodings)})")
        
        return True
    except Exception as e:
        print(f"❌ Error en funcionalidad básica: {e}")
        return False

def test_dlib_backend():
    """Prueba que dlib (backend de face_recognition) funciona."""
    try:
        import dlib
        print("✅ dlib importado correctamente")
        
        # Verificar que los modelos están disponibles
        try:
            detector = dlib.get_frontal_face_detector()
            print("✅ Detector de rostros HOG cargado")
        except Exception as e:
            print(f"⚠️ Advertencia con detector HOG: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Error importando dlib: {e}")
        print("💡 Instala con: pip install dlib")
        return False

def main():
    """Ejecuta todas las pruebas."""
    print("🧪 Probando migración a face_recognition real...")
    print("=" * 50)
    
    tests = [
        ("Importación de face_recognition", test_face_recognition_import),
        ("Importación de dlib", test_dlib_backend),
        ("Funcionalidad básica", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Migración exitosa! face_recognition está listo para usar.")
        print("🚀 Puedes hacer deploy a Streamlit Cloud.")
    else:
        print("⚠️ Hay problemas con la instalación.")
        print("💡 Revisa las dependencias antes de hacer deploy.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
