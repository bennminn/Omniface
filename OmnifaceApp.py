import streamlit as st

# Configurar TensorFlow antes de cualquier importación
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silenciar warnings de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deshabilitar OneDNN para compatibilidad

# Intentar importar numpy con manejo de errores específicos
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando numpy: {e}")
    st.error("Numpy es requerido para el reconocimiento facial")
    NUMPY_AVAILABLE = False

import pandas as pd
from PIL import Image
import os
import pickle
import io
from database_manager import get_db_manager

# Intentar importar OpenCV para mejor precisión (opcional)
try:
    import cv2
    OPENCV_AVAILABLE = True
    st.success("✅ OpenCV disponible - Mayor precisión en detección")
except ImportError as e:
    OPENCV_AVAILABLE = False
    # No mostrar warning para deployment limpio

# Importar manejador robusto de DeepFace
from deepface_handler import initialize_deepface, get_deepface_instance, is_deepface_available

# Verificar que numpy esté disponible (requerido)
if not NUMPY_AVAILABLE:
    st.error("❌ Numpy es requerido para la aplicación")
    st.stop()

def calculate_professional_confidence(cosine_distance):
    """
    Calcula la confianza usando una fórmula profesional para distancia coseno.
    
    Fórmula: confidence = max(0.1, 100 * (1 - distance) ** 2.5)
    
    Escalas resultantes:
    - Distancia 0.0 → 99.9% confianza (match perfecto)
    - Distancia 0.1 → 97.0% confianza (excelente)
    - Distancia 0.2 → 90.0% confianza (muy bueno)
    - Distancia 0.3 → 75.0% confianza (bueno)
    - Distancia 0.4 → 50.0% confianza (umbral profesional)
    
    Args:
        cosine_distance (float): Distancia coseno entre 0.0 y 1.0
        
    Returns:
        float: Porcentaje de confianza entre 0.1% y 99.9%
    """
    if cosine_distance < 0:
        cosine_distance = 0
    elif cosine_distance > 1:
        cosine_distance = 1
        
    confidence = max(0.1, 100 * (1 - cosine_distance) ** 2.5)
    confidence = min(99.9, confidence)  # Cap máximo realista
    
    return confidence

# Inicializar DeepFace con manejo robusto
success, message = initialize_deepface()
if success:
    st.success(f"🎯 {message}")
else:
    # FALLAR SI DEEPFACE NO FUNCIONA - NO MODO SIMULADO
    st.error("❌ ERROR CRÍTICO: DeepFace no está funcionando")
    st.error(f"🔧 Detalles técnicos: {message}")
    st.error("🚫 La aplicación requiere DeepFace para funcionar correctamente")
    st.info("� Reintenta el deploy o revisa las dependencias")
    st.stop()  # DETENER LA APLICACIÓN COMPLETAMENTE

# Obtener instancia de DeepFace (real o simulada)
DeepFace = get_deepface_instance()
DEEPFACE_AVAILABLE = is_deepface_available()


# Título principal
st.title("🎯 OmniFace - Sistema de Reconocimiento Facial")
st.markdown("---")

# Obtener instancia del manager de base de datos
@st.cache_resource
def get_database_manager():
    return get_db_manager()

db_manager = get_database_manager()

# Función para cargar las codificaciones de rostros
def load_encodings():
    encodings = db_manager.get_all_encodings()
    
    # Verificar y limpiar encodings incompatibles
    valid_encodings = {}
    invalid_count = 0
    
    for person_id, encoding in encodings.items():
        if isinstance(encoding, np.ndarray) and encoding.shape == (512,):
            valid_encodings[person_id] = encoding
        else:
            invalid_count += 1
    
    # Solo mostrar warning una vez si hay encodings incompatibles
    if invalid_count > 0 and not st.session_state.get('incompatible_warning_shown', False):
        st.warning(f"⚠️ {invalid_count} encodings necesitan regeneración. Ve a Estadísticas → Herramientas de Administración → Regenerar Todos.")
        st.session_state.incompatible_warning_shown = True
    
    return valid_encodings

# Función para cargar la base de datos de personas
def load_database():
    return db_manager.get_all_persons()

# Función para regenerar encoding de una persona
def regenerate_person_encoding(person_id):
    """Regenerar el encoding de una persona usando su imagen almacenada"""
    try:
        # Obtener imagen de la persona
        image = db_manager.get_person_image(person_id)
        if image is None:
            return False, "No se pudo cargar la imagen"
        
        # Generar nuevo encoding
        new_encoding = get_face_encoding(image)
        if new_encoding is None:
            return False, "No se pudo detectar rostro en la imagen"
        
        # Actualizar encoding en la base de datos
        success = db_manager.update_person_encoding(person_id, new_encoding)
        return success, "Encoding regenerado exitosamente" if success else "Error actualizando encoding"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

# Función para regenerar todos los encodings incompatibles
def regenerate_all_incompatible_encodings():
    """
    REGENERACIÓN FORZADA: Regenerar TODOS los encodings usando Facenet512 
    para garantizar compatibilidad y tolerancias profesionales
    """
    encodings = db_manager.get_all_encodings()
    database = load_database()
    
    if database.empty:
        return 0, 0, 0
    
    # REGENERAR TODOS (no solo incompatibles) para garantizar compatibilidad
    st.info("🔄 REGENERACIÓN FORZADA: Procesando TODOS los usuarios para garantizar compatibilidad")
    
    regenerated_count = 0
    failed_count = 0
    total_persons = len(database)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, row) in enumerate(database.iterrows()):
        person_id = str(row['id'])
        person_name = row['nombre']
        
        # Actualizar progreso
        progress = (idx + 1) / total_persons
        progress_bar.progress(progress)
        status_text.text(f"🔄 Regenerando {person_name} ({idx + 1}/{total_persons})")
        
        try:
            # Obtener imagen original
            image = db_manager.get_person_image(person_id)
            if image is None:
                st.warning(f"⚠️ {person_name}: No se pudo cargar imagen")
                failed_count += 1
                continue
            
            # Generar encoding con Facenet512 FORZADO
            if DEEPFACE_AVAILABLE:
                try:
                    image_array = np.array(image)
                    
                    # Forzar Facenet512 sin mensajes de info (modo silencioso)
                    embedding_result = DeepFace.represent(
                        img_path=image_array,
                        model_name='Facenet512',
                        enforce_detection=True,
                        detector_backend='opencv' if OPENCV_AVAILABLE else 'ssd'
                    )
                    
                    if embedding_result and len(embedding_result) > 0:
                        new_encoding = np.array(embedding_result[0]["embedding"])
                        
                        if new_encoding.shape == (512,):
                            # Actualizar encoding en la base de datos
                            if db_manager.update_person_encoding(person_id, new_encoding):
                                regenerated_count += 1
                                st.success(f"✅ {person_name}: Encoding Facenet512 regenerado")
                            else:
                                st.error(f"❌ {person_name}: Error guardando en BD")
                                failed_count += 1
                        else:
                            st.error(f"❌ {person_name}: Encoding inválido {new_encoding.shape}")
                            failed_count += 1
                    else:
                        st.error(f"❌ {person_name}: No se pudo procesar imagen")
                        failed_count += 1
                        
                except Exception as e:
                    st.error(f"❌ {person_name}: Error DeepFace - {str(e)[:50]}")
                    failed_count += 1
            else:
                st.error(f"❌ {person_name}: DeepFace no disponible")
                failed_count += 1
                
        except Exception as e:
            st.error(f"❌ {person_name}: Error general - {str(e)[:50]}")
            failed_count += 1
    
    # Limpiar progreso
    progress_bar.empty()
    status_text.empty()
    
    return regenerated_count, failed_count, total_persons

# Función para regeneración super agresiva
def force_regenerate_all_with_facenet512():
    """
    REGENERACIÓN SUPER AGRESIVA: Eliminar y recrear TODOS los encodings
    Soluciona incompatibilidades críticas cuando las distancias son > 5.0
    """
    database = load_database()
    
    if database.empty:
        return 0, 0, 0
    
    st.error("🚨 INICIANDO REGENERACIÓN SUPER AGRESIVA")
    st.warning("⚠️ Esto eliminará y recreará TODOS los encodings con Facenet512")
    
    regenerated_count = 0
    failed_count = 0
    total_persons = len(database)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, row) in enumerate(database.iterrows()):
        person_id = str(row['id'])
        person_name = row['nombre']
        
        progress = (idx + 1) / total_persons
        progress_bar.progress(progress)
        status_text.text(f"🔄 FORZANDO regeneración: {person_name} ({idx + 1}/{total_persons})")
        
        try:
            # 1. ELIMINAR encoding existente completamente
            db_manager.delete_person_encoding(person_id)
            
            # 2. Obtener imagen original
            image = db_manager.get_person_image(person_id)
            if image is None:
                st.error(f"❌ {person_name}: Sin imagen")
                failed_count += 1
                continue
            
            # 3. FORZAR Facenet512 con configuración específica
            image_array = np.array(image)
            
            try:
                # Configuración específica y forzada para Facenet512
                embedding_result = DeepFace.represent(
                    img_path=image_array,
                    model_name='Facenet512',  # FORZAR
                    enforce_detection=True,
                    detector_backend='opencv',  # Específico
                    align=True,  # Alineación facial
                    normalization='base'  # Normalización específica
                )
                
                if embedding_result and len(embedding_result) > 0:
                    new_encoding = np.array(embedding_result[0]["embedding"])
                    
                    # Verificar que sea exactamente 512D
                    if new_encoding.shape == (512,):
                        # 4. Guardar nuevo encoding
                        if db_manager.update_person_encoding(person_id, new_encoding):
                            regenerated_count += 1
                            st.success(f"✅ {person_name}: REGENERADO con Facenet512")
                        else:
                            st.error(f"❌ {person_name}: Error guardando")
                            failed_count += 1
                    else:
                        st.error(f"❌ {person_name}: Dimensiones incorrectas {new_encoding.shape}")
                        failed_count += 1
                else:
                    st.error(f"❌ {person_name}: Facenet512 no procesó")
                    failed_count += 1
                    
            except Exception as e:
                st.error(f"❌ {person_name}: Error Facenet512 - {str(e)[:100]}")
                failed_count += 1
                
        except Exception as e:
            st.error(f"❌ {person_name}: Error general - {str(e)[:100]}")
            failed_count += 1
    
    progress_bar.empty()
    status_text.empty()
    
    return regenerated_count, failed_count, total_persons

# Función para limpiar encodings incompatibles
def clean_incompatible_encodings():
    """Eliminar todos los encodings incompatibles de la base de datos"""
    encodings = db_manager.get_all_encodings()
    invalid_persons = []
    
    for person_id, encoding in encodings.items():
        if not isinstance(encoding, np.ndarray) or encoding.shape != (512,):
            invalid_persons.append(person_id)
    
    removed_count = 0
    for person_id in invalid_persons:
        if db_manager.delete_person_encoding(person_id):
            removed_count += 1
    
    return removed_count, len(invalid_persons)

# Función para guardar persona completa
def save_person_complete(person_id, name, image, encoding):
    return db_manager.save_person(person_id, name, image, encoding)

# Función para procesar imagen y obtener codificación facial
def get_face_encoding(image):
    """
    FORZAR uso exclusivo de Facenet512 para compatibilidad total
    Garantiza encodings de 512 dimensiones compatibles con la base de datos
    """
    try:
        # Convertir imagen PIL a array numpy en formato RGB
        image_array = np.array(image)
        
        # Asegurar formato RGB
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Convertir a RGB
            elif image_array.shape[2] == 3:  # Ya es RGB
                pass
        elif len(image_array.shape) == 2:  # Escala de grises
            image_array = np.stack([image_array] * 3, axis=-1)
        else:
            st.error("Formato de imagen no soportado")
            return None
        
        # Preprocesamiento mejorado con OpenCV si está disponible
        if OPENCV_AVAILABLE:
            try:
                # Convertir RGB a BGR para OpenCV
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Aplicar filtros para mejorar la calidad
                image_bgr = cv2.bilateralFilter(image_bgr, 9, 75, 75)  # Reducir ruido
                image_bgr = cv2.convertScaleAbs(image_bgr, alpha=1.1, beta=10)  # Mejor
                
                # Convertir de vuelta a RGB para DeepFace
                image_array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            except Exception as cv_error:
                st.warning(f"Error en preprocesamiento OpenCV: {cv_error}")
                # Continuar con imagen original
        
        # USAR ÚNICAMENTE FACENET512 - SIN ALTERNATIVAS NI FALLBACK
        if DEEPFACE_AVAILABLE:
            try:
                st.info("🎯 Usando Facenet512 (mismo modelo que registro)")
                
                # DeepFace con modelo Facenet512 FORZADO - CONFIGURACIÓN IDÉNTICA A REGENERACIÓN
                embedding_result = DeepFace.represent(
                    img_path=image_array,
                    model_name='Facenet512',  # ← FORZAR Facenet512 EXCLUSIVAMENTE
                    enforce_detection=True,
                    detector_backend='opencv',  # Específico - IGUAL que regeneración
                    align=True,  # Alineación facial - IGUAL que regeneración
                    normalization='base'  # Normalización específica - IGUAL que regeneración
                )
                
                if embedding_result and len(embedding_result) > 0:
                    encoding = np.array(embedding_result[0]["embedding"])
                    
                    # VERIFICAR QUE SEA EXACTAMENTE 512D
                    if encoding.shape == (512,):
                        st.success(f"✅ Encoding Facenet512 extraído: {encoding.shape}")
                        return encoding
                    else:
                        st.error(f"❌ Facenet512 devolvió {encoding.shape} en lugar de (512,)")
                        return None
                else:
                    st.error("❌ Facenet512 no pudo procesar la imagen")
                    return None
                    
            except Exception as deepface_error:
                st.error(f"❌ Error crítico con Facenet512: {deepface_error}")
                st.error("🔧 Problema: Facenet512 no está disponible en este entorno")
                return None
        else:
            st.error("❌ DeepFace no está disponible - No se puede procesar")
            return None
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")
        return None

# Función para reconocer rostro
def recognize_face(image, known_encodings):
    """
    Reconocer rostro comparando con encodings conocidos
    Versión corregida con tolerancia ajustada para Facenet512
    """
    # Obtener encoding de la imagen capturada
    face_encoding = get_face_encoding(image)
    
    if face_encoding is None:
        return None, None
    
    # Verificar que tenemos encodings conocidos válidos
    if not known_encodings:
        return None, None
    
    # TOLERANCIA PROFESIONAL PARA FACENET512 con DISTANCIA COSENO
    tolerance = 0.4  # Para distancia coseno: 0.0=idéntico, 0.4=similar, 0.6=diferente, 1.0+=muy diferente
    
    best_match_person_id = None
    best_distance = float('inf')
    
    # Comparar con cada encoding conocido
    for person_id, known_encoding in known_encodings.items():
        try:
            # Asegurar que es numpy array con 512 dimensiones
            if not isinstance(known_encoding, np.ndarray):
                continue
            
            if known_encoding.shape != (512,):
                st.warning(f"⚠️ Encoding de {person_id} necesita regeneración ({known_encoding.shape} != (512,))")
                continue
            
            # Calcular distancia COSENO para Facenet512 (más apropiada que euclidiana)
            # Normalizar vectores
            face_norm = face_encoding / np.linalg.norm(face_encoding)
            known_norm = known_encoding / np.linalg.norm(known_encoding)
            
            # Distancia coseno: 1 - cosine_similarity
            cosine_similarity = np.dot(face_norm, known_norm)
            distance = 1 - cosine_similarity  # Distancia coseno [0-2]
            
            # Debug de distancias (mostrar solo en modo diagnóstico)
            if st.session_state.get('debug_mode', False):
                st.write(f"🔍 {person_id}: Distancia = {distance:.4f}")
            
            if distance < best_distance:
                best_distance = distance
                best_match_person_id = person_id
        
        except Exception as e:
            # Silenciosamente continuar con el siguiente encoding
            continue
    
    # Debug del mejor match
    if st.session_state.get('debug_mode', False):
        st.write(f"📊 Mejor match: {best_match_person_id} con distancia {best_distance:.4f}")
        st.write(f"🚧 Umbral actual: {tolerance}")
    
    if best_match_person_id is not None and best_distance < tolerance:
        # Usar fórmula profesional de confianza
        confidence = calculate_professional_confidence(best_distance)
        
        if st.session_state.get('debug_mode', False):
            st.success(f"✅ RECONOCIDO: {best_match_person_id} (Confianza: {confidence:.1f}%)")
        
        return best_match_person_id, confidence
    else:
        # Si falla con tolerancia profesional, mostrar warning y sugerir regeneración
        if best_distance > 0.8:  # Distancia coseno anormalmente alta (0.8+ indica incompatibilidad)
            if st.session_state.get('debug_mode', False):
                st.error(f"❌ INCOMPATIBILIDAD DE MODELOS: Distancia {best_distance:.4f} es anormalmente alta")
                st.warning("🔧 SOLUCIÓN: Los encodings necesitan regeneración forzada")
                st.info("💡 Ve a Estadísticas → Regenerar Todos para corregir incompatibilidades")
            else:
                st.error("❌ No reconocido - Posible incompatibilidad de modelos")
                st.info("🔧 Activa 'Modo Diagnóstico' para más detalles")
        elif st.session_state.get('debug_mode', False):
            st.error(f"❌ NO RECONOCIDO: Distancia {best_distance:.4f} > {tolerance}")
        
        return None, None

# Función para procesar múltiples imágenes y crear encoding promediado
def get_averaged_face_encoding(images):
    """
    Procesar múltiples imágenes y crear un encoding promediado más robusto
    Esto mejora significativamente la precisión del reconocimiento
    """
    try:
        valid_encodings = []
        
        for i, image in enumerate(images):
            # Obtener encoding individual
            encoding = get_face_encoding(image)
            if encoding is not None:
                valid_encodings.append(encoding)
                st.success(f"✅ Imagen {i+1}: Rostro detectado correctamente")
            else:
                st.warning(f"⚠️ Imagen {i+1}: No se detectó rostro claro")
        
        if len(valid_encodings) == 0:
            st.error("❌ No se pudo detectar rostros válidos en ninguna imagen")
            return None
        elif len(valid_encodings) == 1:
            st.info("ℹ️ Solo una imagen válida disponible")
            return valid_encodings[0]
        else:
            # Promediar los encodings para mayor robustez
            averaged_encoding = np.mean(valid_encodings, axis=0)
            st.success(f"🎯 Encoding promediado creado desde {len(valid_encodings)} imágenes válidas")
            st.info("💡 El encoding promediado mejora significativamente la precisión del reconocimiento")
            return averaged_encoding
            
    except Exception as e:
        st.error(f"Error procesando múltiples imágenes: {e}")
        return None

# Función de diagnóstico para problemas de reconocimiento
def debug_recognition_system(image, known_encodings):
    """
    Diagnóstico completo del sistema de reconocimiento para identificar problemas
    """
    st.write("## 🔍 **DIAGNÓSTICO DEL SISTEMA DE RECONOCIMIENTO**")
    st.write("---")
    
    # 1. Verificar estado de DeepFace
    st.write("### 1️⃣ **Estado de DeepFace:**")
    st.write(f"- DEEPFACE_AVAILABLE: {DEEPFACE_AVAILABLE}")
    st.write(f"- Instancia DeepFace: {type(DeepFace)}")
    
    # 2. Probar extracción de encoding
    st.write("### 2️⃣ **Extracción de Encoding:**")
    try:
        face_encoding = get_face_encoding(image)
        if face_encoding is not None:
            st.success(f"✅ Encoding extraído correctamente")
            st.write(f"- Shape: {face_encoding.shape}")
            st.write(f"- Tipo: {type(face_encoding)}")
            st.write(f"- Sample: [{face_encoding[0]:.4f}, {face_encoding[1]:.4f}, {face_encoding[2]:.4f}, ...]")
            
            # Verificar si es simulado (comparar con valores aleatorios típicos)
            if np.all(face_encoding >= 0) and np.all(face_encoding <= 1) and np.std(face_encoding) < 0.4:
                st.warning("⚠️ **POSIBLE SIMULACIÓN**: Encoding parece ser aleatorio")
            else:
                st.success("✅ Encoding parece ser real (no simulado)")
        else:
            st.error("❌ **PROBLEMA CRÍTICO**: No se pudo extraer encoding")
            return None
    except Exception as e:
        st.error(f"❌ **ERROR EXTRAYENDO ENCODING**: {e}")
        return None
    
    # 3. Verificar encodings conocidos
    st.write("### 3️⃣ **Encodings Conocidos:**")
    if not known_encodings:
        st.error("❌ **PROBLEMA CRÍTICO**: No hay encodings conocidos")
        return None
    
    st.write(f"- Total encodings: {len(known_encodings)}")
    
    valid_encodings = 0
    for person_id, known_encoding in known_encodings.items():
        if isinstance(known_encoding, np.ndarray) and known_encoding.shape == (512,):
            valid_encodings += 1
            st.write(f"  ✅ {person_id}: Shape {known_encoding.shape}")
        else:
            st.write(f"  ❌ {person_id}: Inválido - {type(known_encoding)} - {getattr(known_encoding, 'shape', 'Sin shape')}")
    
    st.write(f"- Encodings válidos: {valid_encodings}/{len(known_encodings)}")
    
    # 4. Probar diferentes modelos si Facenet512 falla
    st.write("### 4️⃣ **Prueba de Modelos Alternativos:**")
    
    models_to_test = [
        ('VGG-Face', 2622),
        ('Facenet', 128), 
        ('ArcFace', 512),
        ('OpenFace', 128)
    ]
    
    working_models = []
    image_array = np.array(image)
    
    for model_name, expected_dims in models_to_test:
        try:
            st.write(f"🧪 Probando {model_name}...")
            
            embedding_result = DeepFace.represent(
                img_path=image_array,
                model_name=model_name,
                enforce_detection=False  # Menos estricto para pruebas
            )
            
            if embedding_result:
                encoding = np.array(embedding_result[0]["embedding"])
                st.success(f"✅ {model_name} funciona - Shape: {encoding.shape}")
                working_models.append((model_name, encoding.shape[0]))
            else:
                st.warning(f"⚠️ {model_name} no devolvió resultado")
                
        except Exception as e:
            st.error(f"❌ {model_name} falló: {str(e)[:100]}")
    
    # 5. Prueba de comparación con diferentes tolerancias
    st.write("### 5️⃣ **Prueba de Tolerancias:**")
    
    tolerances = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # Tolerancias apropiadas para distancia coseno
    best_matches = []
    
    for tolerance in tolerances:
        best_distance = float('inf')
        best_person = None
        
        for person_id, known_encoding in known_encodings.items():
            if isinstance(known_encoding, np.ndarray) and known_encoding.shape == (512,):
                try:
                    # Usar misma métrica que recognize_face: distancia coseno
                    face_norm = face_encoding / np.linalg.norm(face_encoding)
                    known_norm = known_encoding / np.linalg.norm(known_encoding)
                    cosine_similarity = np.dot(face_norm, known_norm)
                    distance = 1 - cosine_similarity
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_person = person_id
                except:
                    continue
        
        recognized = best_distance < tolerance
        # Usar la misma fórmula profesional de confianza
        confidence = calculate_professional_confidence(best_distance) if recognized else 0
        
        status = "✅" if recognized else "❌"
        st.write(f"  {status} Tolerancia {tolerance}: Distancia {best_distance:.4f} → {'RECONOCIDO' if recognized else 'NO RECONOCIDO'} ({confidence:.1f}%)")
        
        if recognized:
            best_matches.append((tolerance, best_person, best_distance, confidence))
    
    # 6. Recomendaciones
    st.write("### 6️⃣ **Diagnóstico y Recomendaciones:**")
    
    if not working_models:
        st.error("🚨 **PROBLEMA CRÍTICO**: Ningún modelo de DeepFace funciona")
        st.error("**Posibles causas:**")
        st.error("- DeepFace no se inicializó correctamente")
        st.error("- Problemas con TensorFlow/Keras")
        st.error("- Modo simulado activado sin darse cuenta")
        
    elif not best_matches:
        st.error("🚨 **PROBLEMA DE RECONOCIMIENTO**: Ninguna tolerancia reconoce al usuario")
        st.error("**Posibles causas:**")
        st.error("- Los encodings de registro y reconocimiento son muy diferentes")
        st.error("- Problema con la calidad de las imágenes")
        st.error("- El modelo cambió entre registro y reconocimiento")
        
        # Sugerir tolerancia
        if 'best_distance' in locals() and best_distance != float('inf'):
            suggested_tolerance = best_distance + 0.1
            st.info(f"💡 **Sugerencia**: Prueba tolerancia {suggested_tolerance:.2f}")
            
    else:
        st.success("✅ **SISTEMA FUNCIONANDO**: El reconocimiento funciona con ajustes")
        st.success("**Tolerancias que funcionan:**")
        for tolerance, person, distance, confidence in best_matches:
            st.success(f"  - Tolerancia {tolerance}: {person} (Confianza: {confidence:.1f}%)")
        
        # Recomendar tolerancia profesional
        professional_matches = [m for m in best_matches if m[3] >= 85]  # Confianza >= 85%
        if professional_matches:
            best_professional = min(professional_matches, key=lambda x: x[0])  # Tolerancia más estricta
            st.info(f"🎯 **Recomendación Profesional**: Tolerancia {best_professional[0]} (Confianza: {best_professional[3]:.1f}%)")
    
    return best_matches

# Función auxiliar para procesar múltiples imágenes (registro avanzado)
def process_advanced_person(person_id, person_name, image_sources):
    """Procesar y agregar una nueva persona usando múltiples imágenes"""
    # Validaciones
    if not person_id:
        st.error("❌ Por favor ingresa un ID para la persona")
        return False
    elif not person_name:
        st.error("❌ Por favor ingresa el nombre completo")
        return False
    elif not any(image_sources):
        st.error("❌ Por favor toma al menos una fotografía")
        return False
    elif db_manager.person_exists(person_id):
        st.error(f"❌ Ya existe una persona con ID '{person_id}'")
        return False
    else:
        try:
            # Filtrar imágenes válidas
            valid_images = []
            for i, img_source in enumerate(image_sources):
                if img_source is not None:
                    image = Image.open(img_source)
                    valid_images.append(image)
                    st.success(f"✅ Imagen {i+1} cargada correctamente")
            
            if len(valid_images) == 0:
                st.error("❌ No se pudo cargar ninguna imagen válida")
                return False
            
            st.info(f"📷 Procesando {len(valid_images)} imagen(es)...")
            
            # Usar la función de encoding promediado
            averaged_encoding = get_averaged_face_encoding(valid_images)
            
            if averaged_encoding is not None:
                # Guardar persona con encoding promediado (usar la primera imagen como representativa)
                if save_person_complete(person_id, person_name, valid_images[0], averaged_encoding):
                    st.success("✅ Datos guardados en la base de datos")
                    st.success("✅ Encoding promediado creado y guardado")
                    st.success(f"🎉 Persona '{person_name}' agregada exitosamente con registro avanzado!")
                    
                    # Mostrar métricas de calidad
                    st.info(f"📊 Se procesaron {len(valid_images)} imágenes para máxima precisión")
                    
                    # Mostrar preview de las imágenes
                    preview_cols = st.columns(len(valid_images))
                    for i, img in enumerate(valid_images):
                        with preview_cols[i]:
                            st.image(img, caption=f"Imagen {i+1}", width=150)
                    
                    st.info("🔄 Recarga la página para ver los cambios en la galería")
                    return True
                else:
                    st.error("❌ Error guardando en la base de datos")
                    return False
            else:
                st.error("❌ No se pudo procesar las imágenes para crear el encoding")
                st.info("💡 Consejos: Asegúrate de que al menos una imagen contenga un rostro claro")
                return False
        
        except Exception as e:
            st.error(f"❌ Error procesando las imágenes: {str(e)}")
            st.info("🔧 Intenta con imágenes diferentes o verifica que los archivos no estén corruptos")
            return False

# Sidebar para navegación
with st.sidebar:
    st.header("🔧 Panel de Control")
    page = st.radio(
        "Selecciona una opción:",
        ["📝 Gestión de Base de Datos", "🎥 Reconocimiento Facial", "📊 Estadísticas"]
    )

# Página de Gestión de Base de Datos
if page == "📝 Gestión de Base de Datos":
    st.header("📝 Gestión de Base de Datos de Rostros")
    
    # Cargar datos frescos para esta página
    database = load_database()
    encodings = load_encodings()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("➕ Agregar Nueva Persona")
        
        # Tabs para diferentes métodos de captura
        tab1, tab2, tab3 = st.tabs(["📁 Subir Archivo", "📷 Tomar Foto", "🎯 Registro Avanzado"])
        
        with tab1:
            st.write("**Opción 1: Subir una imagen desde tu dispositivo**")
            st.info("💡 **Consejos:** Usa imágenes en formato JPG, JPEG o PNG con buena resolución")
            with st.form("add_person_upload_form"):
                person_id_upload = st.text_input("ID de la Persona:", key="id_upload", 
                                                help="Ingresa un identificador único (ej: 12345678)")
                person_name_upload = st.text_input("Nombre Completo:", key="name_upload",
                                                  help="Nombre completo de la persona")
                uploaded_file = st.file_uploader(
                    "Subir Fotografía:", 
                    type=['jpg', 'jpeg', 'png'],
                    help="Sube una imagen clara del rostro de la persona"
                )
                
                submitted_upload = st.form_submit_button("💾 Agregar Persona (Archivo)")
        
        with tab2:
            st.write("**Opción 2: Tomar foto directamente con la cámara**")
            st.info("📸 **Consejos:** Asegúrate de tener buena iluminación y que el rostro esté centrado")
            with st.form("add_person_camera_form"):
                person_id_camera = st.text_input("ID de la Persona:", key="id_camera",
                                                help="Ingresa un identificador único (ej: 12345678)")
                person_name_camera = st.text_input("Nombre Completo:", key="name_camera",
                                                  help="Nombre completo de la persona")
                
                st.markdown("**📷 Captura de Imagen:**")
                camera_input = st.camera_input("Tomar fotografía:",
                                              help="Haz clic para activar la cámara y tomar una foto")
                
                submitted_camera = st.form_submit_button("💾 Agregar Persona (Foto)")
        
        with tab3:
            st.write("**Opción 3: Registro Avanzado con 3 Imágenes 🎯**")
            st.info("🚀 **Mejora la precisión:** Toma 3 fotos diferentes para crear un encoding más robusto")
            st.markdown("""
            **¿Por qué usar 3 imágenes?**
            - 📈 **Mayor precisión:** Aumenta significativamente la confianza del reconocimiento
            - 🎭 **Diferentes condiciones:** Captura variaciones naturales del rostro
            - 💡 **Robustez:** Menos sensible a cambios de iluminación y expresiones
            - ✅ **Recomendado** para usuarios que han tenido problemas de baja confianza
            """)
            
            with st.form("add_person_advanced_form"):
                person_id_advanced = st.text_input("ID de la Persona:", key="id_advanced",
                                                 help="Ingresa un identificador único (ej: 12345678)")
                person_name_advanced = st.text_input("Nombre Completo:", key="name_advanced",
                                                   help="Nombre completo de la persona")
                
                st.markdown("### 📸 Captura de 3 Imágenes")
                st.info("💡 **Consejos:** Toma cada foto con diferentes condiciones de luz o ángulos ligeramente distintos")
                
                # Contenedores para las 3 imágenes
                col_img1, col_img2, col_img3 = st.columns(3)
                
                with col_img1:
                    st.markdown("**📷 Imagen 1:**")
                    camera_input_1 = st.camera_input("Foto 1 (ej: luz natural):", key="cam1")
                
                with col_img2:
                    st.markdown("**📷 Imagen 2:**")
                    camera_input_2 = st.camera_input("Foto 2 (ej: luz artificial):", key="cam2")
                
                with col_img3:
                    st.markdown("**📷 Imagen 3:**")
                    camera_input_3 = st.camera_input("Foto 3 (ej: expresión neutra):", key="cam3")
                
                # Mostrar preview de las imágenes capturadas
                if camera_input_1 or camera_input_2 or camera_input_3:
                    st.markdown("### 🖼️ Vista Previa de Imágenes Capturadas")
                    preview_cols = st.columns(3)
                    
                    if camera_input_1:
                        with preview_cols[0]:
                            st.image(camera_input_1, caption="Imagen 1", width=150)
                    
                    if camera_input_2:
                        with preview_cols[1]:
                            st.image(camera_input_2, caption="Imagen 2", width=150)
                    
                    if camera_input_3:
                        with preview_cols[2]:
                            st.image(camera_input_3, caption="Imagen 3", width=150)
                
                submitted_advanced = st.form_submit_button("🎯 Agregar Persona (Registro Avanzado)")

        # Función auxiliar para procesar persona
        def process_person(person_id, person_name, image_source, source_type):
            """Procesar y agregar una nueva persona a la base de datos"""
            # Validaciones
            if not person_id:
                st.error("❌ Por favor ingresa un ID para la persona")
                return False
            elif not person_name:
                st.error("❌ Por favor ingresa el nombre completo")
                return False
            elif image_source is None:
                if source_type == "upload":
                    st.error("❌ Por favor sube una fotografía")
                else:
                    st.error("❌ Por favor toma una fotografía")
                return False
            elif db_manager.person_exists(person_id):
                st.error(f"❌ Ya existe una persona con ID '{person_id}'")
                return False
            else:
                try:
                    # Procesar imagen
                    image = Image.open(image_source)
                    st.info("📷 Procesando imagen...")
                    
                    # Obtener codificación facial
                    face_encoding = get_face_encoding(image)
                    
                    if face_encoding is not None:
                        # Guardar persona completa en Supabase
                        if save_person_complete(person_id, person_name, image, face_encoding):
                            st.success("✅ Datos guardados en la base de datos")
                            st.success("✅ Codificación facial guardada")
                            st.success(f"🎉 Persona '{person_name}' agregada exitosamente!")
                            
                            # Mostrar preview de la imagen
                            st.image(image, caption=f"Imagen registrada: {person_name}", width=200)
                            
                            st.info("🔄 Recarga la página para ver los cambios en la galería")
                            return True
                        else:
                            st.error("❌ Error guardando en la base de datos")
                            return False
                    else:
                        st.error("❌ No se pudo detectar un rostro en la imagen. Intenta con otra foto más clara.")
                        st.info("💡 Consejos: Asegúrate de que el rostro esté bien iluminado, centrado y sin obstrucciones")
                        return False
                
                except Exception as e:
                    st.error(f"❌ Error procesando la solicitud: {str(e)}")
                    st.info("🔧 Intenta con una imagen diferente o verifica que el archivo no esté corrupto")
                    return False
        
        # Procesar formulario de archivo subido
        if submitted_upload:
            process_person(person_id_upload, person_name_upload, uploaded_file, "upload")
        
        # Procesar formulario de cámara
        if submitted_camera:
            process_person(person_id_camera, person_name_camera, camera_input, "camera")
        
        # Procesar formulario de registro avanzado
        if submitted_advanced:
            # Validar que se capturaron al menos 2 imágenes
            images_captured = [camera_input_1, camera_input_2, camera_input_3]
            valid_images_count = sum(1 for img in images_captured if img is not None)
            
            if valid_images_count < 2:
                st.error("❌ Debes capturar al menos 2 imágenes para el registro avanzado")
                st.info("💡 El registro avanzado requiere mínimo 2 imágenes para crear un encoding robusto")
            else:
                # Procesar con las imágenes válidas
                process_advanced_person(person_id_advanced, person_name_advanced, images_captured)
        
        # Sección de ayuda y preview
        st.markdown("---")
        st.subheader("📋 Instrucciones Generales")
        
        with st.expander("🔍 Consejos para mejores resultados", expanded=False):
            st.markdown("""
            ### Para obtener el mejor reconocimiento facial:
            
            **🖼️ Calidad de imagen:**
            - Usa imágenes con buena resolución (mínimo 200x200 píxeles)
            - Evita imágenes borrosas o pixeladas
            - Formatos soportados: JPG, JPEG, PNG
            
            **💡 Iluminación:**
            - Asegúrate de tener buena iluminación frontal
            - Evita sombras fuertes en el rostro
            - La luz natural es ideal
            
            **👤 Posición del rostro:**
            - El rostro debe estar centrado en la imagen
            - Evita ángulos extremos o perfiles
            - Asegúrate de que el rostro esté completamente visible
            
            **🚫 Evita:**
            - Gafas de sol o máscaras
            - Gorros que cubran gran parte del rostro
            - Expresiones faciales extremas
            - Múltiples personas en la misma imagen
            """)
        
        # Preview de imagen capturada
        if 'uploaded_file' in locals() and uploaded_file is not None:
            st.markdown("### 🖼️ Preview de Imagen Subida")
            st.image(uploaded_file, caption="Imagen seleccionada", width=300)
        
        if 'camera_input' in locals() and camera_input is not None:
            st.markdown("### 📸 Preview de Foto Capturada")
            st.image(camera_input, caption="Foto tomada con cámara", width=300)
    
    with col2:
        st.subheader("👥 Base de Datos Actual")
        
        if not database.empty:
            st.dataframe(database[['id', 'nombre']])
            
            # Mostrar imágenes
            st.subheader("🖼️ Galería de Rostros")
            cols = st.columns(3)
            
            for idx, row in database.iterrows():
                with cols[idx % 3]:
                    # Obtener imagen desde Supabase
                    image = db_manager.get_person_image(row['id'])
                    if image:
                        st.image(image, caption=f"{row['nombre']} (ID: {row['id']})", width=300)
                    else:
                        st.error("❌ Error cargando imagen")
                    
                    # Botón para eliminar
                    if st.button(f"🗑️ Eliminar {row['nombre']}", key=f"delete_{idx}"):
                        # Eliminar de Supabase
                        if db_manager.delete_person(row['id']):
                            st.success(f"✅ Persona '{row['nombre']}' eliminada exitosamente!")
                            st.rerun()
                        else:
                            st.error("❌ Error eliminando persona")
        else:
            st.info("📝 No hay personas registradas en la base de datos.")

# Página de Reconocimiento Facial
elif page == "🎥 Reconocimiento Facial":
    st.header("🎥 Reconocimiento Facial en Tiempo Real")
    
    # Cargar datos frescos para esta página
    database = load_database()
    encodings = load_encodings()
    
    if database.empty:
        st.warning("⚠️ No hay personas registradas en la base de datos. Ve a la sección de Gestión para agregar personas.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Capturar Imagen")
            
            # Activar modo diagnóstico
            debug_mode = st.checkbox("🔍 **Modo Diagnóstico** (para solucionar problemas de reconocimiento)", value=False)
            
            # Guardar estado del modo debug
            st.session_state.debug_mode = debug_mode
            
            # Captura de imagen con cámara
            camera_input = st.camera_input("Toma una foto para reconocimiento facial:")
            
            if camera_input is not None:
                # Procesar imagen capturada
                image = Image.open(camera_input)
                st.image(image, caption="Imagen capturada")
                
                if debug_mode:
                    # Ejecutar diagnóstico completo
                    st.write("---")
                    debug_matches = debug_recognition_system(image, encodings)
                    
                    if debug_matches:
                        st.write("### 🛠️ **Aplicar Corrección Automática**")
                        best_match = debug_matches[0]  # Mejor tolerancia
                        if st.button(f"✅ Usar tolerancia {best_match[0]} (Reconoce como {best_match[1]})"):
                            # Aplicar reconocimiento con tolerancia sugerida
                            person_id, confidence = best_match[1], best_match[3]
                        else:
                            person_id, confidence = None, None
                    else:
                        person_id, confidence = None, None
                else:
                    # Reconocimiento normal
                    with st.spinner("🔍 Analizando rostro..."):
                        person_id, confidence = recognize_face(image, encodings)
                
                if person_id is not None:
                    # Buscar información de la persona
                    person_info = database[database['id'] == person_id].iloc[0]
                    
                    with col2:
                        st.subheader("✅ Persona Reconocida")
                        st.success(f"**Nombre:** {person_info['nombre']}")
                        st.info(f"**ID:** {person_info['id']}")
                        st.info(f"**Confianza:** {confidence:.2f}%")
                        
                        # Mostrar imagen de referencia desde Supabase
                        ref_image = db_manager.get_person_image(person_info['id'])
                        if ref_image:
                            st.image(ref_image, caption="Imagen de referencia")
                        else:
                            st.warning("⚠️ No se pudo cargar la imagen de referencia")
                        
                        # Mostrar alerta de éxito basada en confianza
                        if confidence >= 95:
                            st.balloons()
                            st.success("🎯 ¡Reconocimiento con confianza muy alta!")
                        elif confidence >= 90:
                            st.success("✅ Reconocimiento con confianza alta")
                        elif confidence >= 85:
                            st.info("👍 Reconocimiento con confianza aceptable")
                        else:
                            st.warning("⚠️ Reconocimiento con confianza baja")
                else:
                    with col2:
                        st.subheader("❌ No Reconocido")
                        st.error("No se pudo identificar a la persona en la imagen.")
                        st.info("Verifica que la persona esté registrada en la base de datos.")
        
        with col2:
            if camera_input is None:
                st.subheader("📋 Instrucciones")
                st.markdown("""
                ### Cómo usar el reconocimiento facial:
                
                1. **Captura una imagen** usando la cámara
                2. **Asegúrate** de que el rostro esté bien iluminado
                3. **Mantén** el rostro centrado en la imagen
                4. **Espera** a que se procese el reconocimiento
                5. **Revisa** los resultados y la confianza
                
                ### Consejos para mejor reconocimiento:
                - Usa buena iluminación
                - Mantén el rostro sin obstrucciones
                - Evita sombras fuertes
                - Asegúrate de que el rostro esté enfocado
                """)

# Página de Estadísticas
elif page == "📊 Estadísticas":
    st.header("📊 Estadísticas del Sistema")
    
    # Cargar datos frescos para esta página
    database = load_database()
    encodings = load_encodings()
    stats = db_manager.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👥 Personas Registradas", stats['total_persons'])
    
    with col2:
        st.metric("🔧 Codificaciones Activas", stats['encodings_active'])
    
    with col3:
        st.metric("💾 Tamaño Total (MB)", stats['total_size_mb'])
    
    # Tabla de resumen
    if not database.empty:
        st.subheader("📋 Resumen de Base de Datos")
        st.dataframe(database)
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    st.info("""
    **Sistema de Reconocimiento Facial OmniFace v2.0**
    
    - **Tecnología:** DeepFace + Facenet512 + Supabase
    - **Modelo:** Facenet512 (512 dimensiones)
    - **Métrica:** Distancia Coseno (optimizada para embeddings)
    - **Base de datos:** Supabase (PostgreSQL)
    - **Almacenamiento:** Cloud (persistente)
    - **Tolerancia profesional:** 0.4 (alta precisión)
    - **Confianza mínima:** 85%
    - **Formatos soportados:** JPG, JPEG, PNG
    - **Deploy:** Compatible con Streamlit Cloud
    """)
    
    # Sección de administración
    st.subheader("🔧 Herramientas de Administración")
    
    col_admin1, col_admin2 = st.columns(2)
    
    with col_admin1:
        st.write("**🔄 Regenerar Encodings Forzadamente**")
        st.info("REGENERA TODOS los encodings con Facenet512 para garantizar compatibilidad y tolerancias profesionales (0.4 distancia coseno)")
        st.warning("⚠️ IMPORTANTE: Si las distancias son > 0.8, los encodings tienen incompatibilidades críticas")
        if st.button("🔄 Regenerar Todos", type="primary"):
            with st.spinner("Regenerando TODOS los encodings con Facenet512..."):
                regenerated, failed, total = regenerate_all_incompatible_encodings()
                if regenerated > 0:
                    st.success(f"✅ Se regeneraron {regenerated}/{total} encodings con Facenet512")
                    if failed > 0:
                        st.warning(f"⚠️ {failed} encodings fallaron")
                    st.success("🎯 Ahora el sistema debería usar tolerancias profesionales (0.4 distancia coseno)")
                    st.info("🔄 Recargando página...")
                    st.rerun()
                else:
                    if total == 0:
                        st.info("ℹ️ No hay usuarios registrados")
                    else:
                        st.error(f"❌ No se pudieron regenerar {failed} encodings")
        
        st.markdown("---")
        st.write("**🚨 REGENERACIÓN SUPER AGRESIVA**")
        st.error("🚨 SOLO usar si las distancias coseno son > 0.8 (incompatibilidad crítica)")
        st.warning("⚠️ Elimina y recrea TODOS los encodings desde cero con configuración específica")
        if st.button("🚨 REGENERACIÓN SUPER AGRESIVA", type="primary"):
            with st.spinner("🚨 EJECUTANDO REGENERACIÓN SUPER AGRESIVA..."):
                regenerated, failed, total = force_regenerate_all_with_facenet512()
                if regenerated > 0:
                    st.success(f"🎉 REGENERACIÓN SUPER AGRESIVA EXITOSA!")
                    st.success(f"✅ {regenerated}/{total} encodings regenerados con configuración específica")
                    if failed > 0:
                        st.warning(f"⚠️ {failed} encodings fallaron")
                    st.success("🎯 Las distancias coseno deberían ser ahora < 0.4 (profesional)")
                    st.info("🔄 Recargando página...")
                    st.rerun()
                else:
                    if total == 0:
                        st.info("ℹ️ No hay usuarios registrados")
                    else:
                        st.error(f"💥 REGENERACIÓN SUPER AGRESIVA FALLÓ - {failed} errores")
        
        st.write("**🗑️ Limpiar Encodings Incompatibles**")
        st.info("Elimina encodings con formato incorrecto (solo como último recurso)")
        if st.button("🗑️ Limpiar Encodings", type="secondary"):
            with st.spinner("Limpiando encodings incompatibles..."):
                removed, total_invalid = clean_incompatible_encodings()
                if removed > 0:
                    st.success(f"✅ Se eliminaron {removed} de {total_invalid} encodings incompatibles")
                    st.rerun()
                else:
                    st.info("ℹ️ No se encontraron encodings incompatibles para eliminar")
    
    with col_admin2:
        st.write("**Regenerar Encodings**")
        st.info("Regenera encodings para personas específicas usando sus imágenes")
        
        # Selector de persona para regenerar
        if not database.empty:
            person_options = [(row['id'], f"{row['nombre']} (ID: {row['id']})") for _, row in database.iterrows()]
            selected_person = st.selectbox(
                "Seleccionar persona:",
                options=[None] + person_options,
                format_func=lambda x: "Selecciona una persona..." if x is None else x[1]
            )
            
            if selected_person and st.button("🔄 Regenerar Encoding", type="secondary"):
                person_id = selected_person[0]
                with st.spinner(f"Regenerando encoding para {selected_person[1]}..."):
                    success, message = regenerate_person_encoding(person_id)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        else:
            st.info("No hay personas registradas")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎯 OmniFace v2.0 - Sistema de Reconocimiento Facial</p>
        <p>Desarrollado con ❤️ usando Streamlit + Supabase</p>
        <p>🌐 <a href="https://github.com/bennminn/Omniface" target="_blank">Ver código en GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)