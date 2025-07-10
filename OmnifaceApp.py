import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle
import io
from database_manager import get_db_manager

# Intentar importar face_recognition con manejo de errores  
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando face_recognition: {e}")
    st.error("Cayendo de vuelta al simulador...")
    FACE_RECOGNITION_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="OmniFace - Reconocimiento Facial",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    return db_manager.get_all_encodings()

# Función para cargar la base de datos de personas
def load_database():
    return db_manager.get_all_persons()

# Función para guardar persona completa
def save_person_complete(person_id, name, image, encoding):
    return db_manager.save_person(person_id, name, image, encoding)

# Función para procesar imagen y obtener codificación facial
def get_face_encoding(image):
    try:
        # Convertir imagen PIL a array numpy
        image_array = np.array(image)
        
        # Verificar que la imagen tiene el formato correcto
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB
            pass
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA - convertir a RGB
            image_array = image_array[:, :, :3]
        elif len(image_array.shape) == 2:
            # Escala de grises - convertir a RGB usando numpy
            image_array = np.stack([image_array] * 3, axis=-1)
        else:
            st.error("Formato de imagen no soportado")
            return None
        
        # Detectar rostros en la imagen
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) == 0:
            return None
        
        # Obtener codificación del primer rostro encontrado
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) > 0:
            return face_encodings[0]
        
        return None
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None

# Función para reconocer rostro
def recognize_face(image, known_encodings):
    face_encoding = get_face_encoding(image)
    
    if face_encoding is None:
        return None, None
    
    # Usar tolerancia más estricta para mayor precisión
    tolerance = 0.25  # Más estricto que 0.4
    min_confidence = 90  # Confianza mínima requerida
    
    best_match = None
    best_confidence = 0
    best_person_id = None
    
    # Comparar con todos los rostros conocidos y encontrar el mejor match
    for person_id, known_encoding in known_encodings.items():
        matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
        
        if matches[0]:
            # Calcular distancia y confianza
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            confidence = max(0, (1 - distance * 2.5) * 100)  # Fórmula ajustada para mayor precisión
            
            # Solo considerar si supera la confianza mínima
            if confidence >= min_confidence and confidence > best_confidence:
                best_confidence = confidence
                best_person_id = person_id
                best_match = True
    
    if best_match:
        return best_person_id, best_confidence
    else:
        return None, None

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
        tab1, tab2 = st.tabs(["📁 Subir Archivo", "📷 Tomar Foto"])
        
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
                
                submitted_upload = st.form_submit_button("💾 Agregar Persona (Archivo)", 
                                                        use_container_width=True)
        
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
                
                submitted_camera = st.form_submit_button("💾 Agregar Persona (Foto)", 
                                                        use_container_width=True)
        
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
            st.dataframe(database[['id', 'nombre']], use_container_width=True)
            
            # Mostrar imágenes
            st.subheader("🖼️ Galería de Rostros")
            cols = st.columns(3)
            
            for idx, row in database.iterrows():
                with cols[idx % 3]:
                    # Obtener imagen desde Supabase
                    image = db_manager.get_person_image(row['id'])
                    if image:
                        st.image(image, caption=f"{row['nombre']} (ID: {row['id']})", use_container_width=True)
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
            
            # Captura de imagen con cámara
            camera_input = st.camera_input("Toma una foto para reconocimiento facial:")
            
            if camera_input is not None:
                # Procesar imagen capturada
                image = Image.open(camera_input)
                st.image(image, caption="Imagen capturada", use_container_width=True)
                
                # Realizar reconocimiento
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
                            st.image(ref_image, caption="Imagen de referencia", use_container_width=True)
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
        st.dataframe(database, use_container_width=True)
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    st.info("""
    **Sistema de Reconocimiento Facial OmniFace v2.0**
    
    - **Tecnología:** OpenCV + Supabase
    - **Base de datos:** Supabase (PostgreSQL)
    - **Almacenamiento:** Cloud (persistente)
    - **Tolerancia:** 0.25 (alta precisión)
    - **Confianza mínima:** 90%
    - **Formatos soportados:** JPG, JPEG, PNG
    - **Deploy:** Compatible con Streamlit Cloud
    """)

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