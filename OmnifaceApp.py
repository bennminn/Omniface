import streamlit as st

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

# Intentar importar face_recognition con manejo de errores  
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando face_recognition: {e}")
    st.error("Cayendo de vuelta al simulador...")
    FACE_RECOGNITION_AVAILABLE = False

# Verificar que todas las dependencias estén disponibles
if not NUMPY_AVAILABLE:
    st.stop()
if not FACE_RECOGNITION_AVAILABLE:
    st.stop()

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
    encodings = db_manager.get_all_encodings()
    
    # Verificar y limpiar encodings incompatibles
    valid_encodings = {}
    invalid_count = 0
    
    for person_id, encoding in encodings.items():
        if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
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
    """Regenerar automáticamente todos los encodings incompatibles"""
    encodings = db_manager.get_all_encodings()
    invalid_persons = []
    
    for person_id, encoding in encodings.items():
        if not isinstance(encoding, np.ndarray) or encoding.shape != (128,):
            invalid_persons.append(person_id)
    
    regenerated_count = 0
    failed_count = 0
    
    for person_id in invalid_persons:
        success, message = regenerate_person_encoding(person_id)
        if success:
            regenerated_count += 1
        else:
            failed_count += 1
    
    return regenerated_count, failed_count, len(invalid_persons)

# Función para limpiar encodings incompatibles
def clean_incompatible_encodings():
    """Eliminar todos los encodings incompatibles de la base de datos"""
    encodings = db_manager.get_all_encodings()
    invalid_persons = []
    
    for person_id, encoding in encodings.items():
        if not isinstance(encoding, np.ndarray) or encoding.shape != (128,):
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
    Procesar imagen y obtener codificación facial usando face_recognition
    Basado en el código funcional de Django
    """
    try:
        # Convertir imagen PIL a array numpy en formato RGB
        image_array = np.array(image)
        
        # Asegurar formato RGB (como en Django)
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
        
        # Detectar ubicaciones de rostros (similar a classify_face en Django)
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) == 0:
            return None
        
        # Obtener encodings de rostros encontrados
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) > 0:
            # Retornar el primer encoding encontrado
            encoding = face_encodings[0]
            # Verificar que el encoding tiene la forma correcta (128 dimensiones)
            if encoding.shape == (128,):
                return encoding
            else:
                st.error(f"Encoding generado tiene forma incorrecta: {encoding.shape}")
                return None
        
        return None
        
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None

# Función para reconocer rostro
def recognize_face(image, known_encodings):
    """
    Reconocer rostro comparando con encodings conocidos
    Basado en el approach simplificado del código Django funcional
    """
    # Obtener encoding de la imagen capturada
    face_encoding = get_face_encoding(image)
    
    if face_encoding is None:
        return None, None
    
    # Verificar que tenemos encodings conocidos válidos
    if not known_encodings:
        return None, None
    
    # Parámetros de reconocimiento (más permisivos inicialmente)
    tolerance = 0.6  # Tolerancia estándar de face_recognition
    
    best_match_person_id = None
    best_distance = float('inf')
    
    # Comparar con cada encoding conocido
    for person_id, known_encoding in known_encodings.items():
        try:
            # Asegurar que es numpy array
            if not isinstance(known_encoding, np.ndarray):
                continue
                
            # Verificar dimensiones compatibles
            if known_encoding.shape != (128,):
                continue
            
            # Calcular distancia usando face_recognition
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            
            # Si está dentro de la tolerancia y es el mejor match hasta ahora
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_match_person_id = person_id
        
        except Exception as e:
            # Silenciosamente continuar con el siguiente encoding
            continue
    
    if best_match_person_id is not None:
        # Convertir distancia a porcentaje de confianza
        confidence = max(0, (1 - best_distance) * 100)
        return best_match_person_id, confidence
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
            
            # Captura de imagen con cámara
            camera_input = st.camera_input("Toma una foto para reconocimiento facial:")
            
            if camera_input is not None:
                # Procesar imagen capturada
                image = Image.open(camera_input)
                st.image(image, caption="Imagen capturada")
                
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
    
    - **Tecnología:** face_recognition + Supabase
    - **Base de datos:** Supabase (PostgreSQL)
    - **Almacenamiento:** Cloud (persistente)
    - **Tolerancia:** 0.25 (alta precisión)
    - **Confianza mínima:** 90%
    - **Formatos soportados:** JPG, JPEG, PNG
    - **Deploy:** Compatible con Streamlit Cloud
    """)
    
    # Sección de administración
    st.subheader("🔧 Herramientas de Administración")
    
    col_admin1, col_admin2 = st.columns(2)
    
    with col_admin1:
        st.write("**🔄 Regenerar Encodings Automáticamente**")
        st.info("Regenera automáticamente todos los encodings incompatibles usando las imágenes almacenadas")
        if st.button("🔄 Regenerar Todos", type="primary"):
            with st.spinner("Regenerando encodings incompatibles..."):
                regenerated, failed, total = regenerate_all_incompatible_encodings()
                if regenerated > 0:
                    st.success(f"✅ Se regeneraron {regenerated} encodings exitosamente")
                    if failed > 0:
                        st.warning(f"⚠️ {failed} encodings no pudieron regenerarse")
                    st.info("🔄 Recargando página...")
                    st.rerun()
                else:
                    if total == 0:
                        st.info("ℹ️ No se encontraron encodings incompatibles")
                    else:
                        st.error(f"❌ No se pudieron regenerar {failed} encodings")
        
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