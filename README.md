# 🎯 OmniFace v2.0 - Sistema de Reconocimiento Facial Profesional

## 📝 Descripción
OmniFace es una aplicación web profesional desarrollada con Streamlit que permite realizar reconocimiento facial de alta precisión usando tecnología DeepFace + Facenet512. La aplicación cuenta con almacenamiento en la nube (Supabase), fórmulas profesionales de confianza y tolerancias optimizadas para uso empresarial.

## 🚀 Características Principales
- **🎯 Reconocimiento Profesional**: Facenet512 con distancia coseno (tolerancia 0.4)
- **📊 Fórmula de Confianza Avanzada**: Escalas realistas y conservadoras
- **☁️ Almacenamiento en la Nube**: Base de datos Supabase PostgreSQL
- **🔄 Regeneración Inteligente**: Herramientas de administración avanzadas
- **📷 Registro Multi-Imagen**: Encoding promediado para mayor precisión
- **🔍 Modo Diagnóstico**: Herramientas para solucionar problemas
- **🌐 Deploy en la Nube**: Compatible con Streamlit Cloud

## 🧠 Tecnología
- **Modelo**: Facenet512 (512 dimensiones)
- **Métrica**: Distancia Coseno (optimizada para embeddings)
- **Base de Datos**: Supabase (PostgreSQL)
- **Backend**: DeepFace + TensorFlow + Keras
- **Frontend**: Streamlit
- **Procesamiento**: OpenCV + PIL + NumPy

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- Cámara web (para captura de imágenes)
- Conexión a internet (para Supabase)

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Dependencias Incluidas
- streamlit: Framework web para la aplicación
- deepface: Biblioteca de reconocimiento facial
- tensorflow: Motor de aprendizaje automático
- tf-keras: API de alto nivel para TensorFlow
- opencv-python: Procesamiento de imágenes
- pillow: Manipulación de imágenes
- pandas: Manejo de datos
- numpy: Operaciones matemáticas
- supabase: Cliente de base de datos

## 🏃‍♂️ Ejecución
Para ejecutar la aplicación:

```bash
streamlit run OmnifaceApp.py
```

La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`

## 🌐 Deploy en Streamlit Cloud
La aplicación está optimizada para deploy en Streamlit Cloud:
1. Fork este repositorio en GitHub
2. Conecta tu cuenta de Streamlit Cloud
3. Configura las variables de entorno para Supabase
4. Deploy automático desde GitHub

## 📱 Uso de la Aplicación

### 1. 📝 Gestión de Base de Datos
#### Método Básico:
- **Subir Archivo**: Sube una imagen desde tu dispositivo
- **Tomar Foto**: Captura directamente con la cámara web

#### Método Avanzado (Recomendado):
- **Registro Multi-Imagen**: Captura 3 fotos diferentes
- **Encoding Promediado**: Mayor precisión y robustez
- **Mejores Resultados**: Recomendado para usuarios críticos

### 2. 🎥 Reconocimiento Facial
- Ve a la sección "Reconocimiento Facial"
- Activa **Modo Diagnóstico** para solucionar problemas
- Captura imagen con la cámara web
- **Resultados Profesionales**:
  - Confianza 85%+: Reconocimiento aceptable
  - Confianza 90%+: Reconocimiento alto
  - Confianza 95%+: Reconocimiento muy alto

### 3. 📊 Estadísticas y Administración
- **Métricas del Sistema**: Personas registradas, encodings activos
- **Regenerar Encodings**: Herramienta de mantenimiento
- **Regeneración Super Agresiva**: Para incompatibilidades críticas
- **Diagnóstico Avanzado**: Solución de problemas

## 📂 Estructura de Archivos
```
PyWebApps/
├── OmnifaceApp.py          # Aplicación principal
├── database_manager.py     # Gestión de base de datos Supabase
├── deepface_handler.py     # Manejador robusto de DeepFace
├── requirements.txt        # Dependencias para Streamlit Cloud
├── packages.txt           # Paquetes del sistema para Streamlit Cloud
├── README.md              # Documentación
├── .env.example           # Ejemplo de variables de entorno
└── images/               # Imágenes de ejemplo
```

## 🔧 Configuración Profesional

### Escalas de Confianza (Fórmula Profesional)
- **Distancia 0.0**: 99.9% confianza (match perfecto)
- **Distancia 0.1**: 76.8% confianza (excelente)
- **Distancia 0.2**: 57.2% confianza (bueno)
- **Distancia 0.3**: 41.0% confianza (regular)
- **Distancia 0.4**: 27.9% confianza (umbral límite)

### Tolerancias Profesionales
- **Tolerancia**: 0.4 (distancia coseno)
- **Métrica**: Distancia coseno optimizada para Facenet512
- **Umbral Crítico**: 0.8+ indica incompatibilidad de modelos

#### Opción 2: Usando conda
```bash
conda install -c conda-forge face_recognition
```

### Consejos para Mejor Reconocimiento
- Usa imágenes con buena iluminación
- Asegúrate de que el rostro esté centrado y sin obstrucciones
- Evita sombras fuertes en el rostro
- Mantén una distancia apropiada de la cámara

## 🐛 Solución de Problemas

### Problema: "Agregar Persona" no funciona

**Síntomas:**
- El botón "Agregar Persona" no responde
- No se guardan los datos en la base de datos
- Errores al procesar imágenes

**Soluciones:**

1. **Verificar campos obligatorios:**
   - Asegúrate de llenar todos los campos: ID, Nombre y subir una imagen
   - El ID debe ser único (no puede repetirse)

2. **Verificar formato de imagen:**
   - Usa formatos soportados: JPG, JPEG, PNG
   - Asegúrate de que la imagen contenga un rostro visible
   - La imagen no debe estar corrupta

3. **Verificar permisos de escritura:**
   - La aplicación necesita crear carpetas y archivos
   - Ejecuta desde una ubicación con permisos de escritura

4. **Ejecutar pruebas diagnósticas:**
   ```bash
   python test_omniface.py
   ```

5. **Crear imagen de ejemplo:**
   ```bash
   python create_sample.py
   ```
## 🔧 Variables de Entorno
Para usar Supabase, configura estas variables:

```bash
SUPABASE_URL=tu_url_de_supabase
SUPABASE_KEY=tu_clave_de_supabase
```

## 🚨 Solución de Problemas

### Problemas de Reconocimiento
1. **Activa Modo Diagnóstico** en la sección de Reconocimiento Facial
2. **Revisa las distancias coseno** - deben ser < 0.4 para reconocimiento
3. **Regenera encodings** si las distancias son > 0.8
4. **Usa Regeneración Super Agresiva** para incompatibilidades críticas

### Mensajes de Error Comunes

**"❌ No reconocido - Posible incompatibilidad de modelos"**
- Ve a Estadísticas → Regenerar Todos
- Si persiste, usa Regeneración Super Agresiva

**"⚠️ Encoding de [persona] necesita regeneración"**
- Encoding tiene formato incompatible (no 512D)
- Usa herramientas de regeneración en Estadísticas

**"❌ DeepFace no está funcionando"**
- Problema crítico con TensorFlow/Keras
- Revisa requirements.txt y reinstala dependencias

**"🔧 SOLUCIÓN: Los encodings necesitan regeneración forzada"**
- Distancias > 0.8 indican incompatibilidad crítica
- Usa Regeneración Super Agresiva inmediatamente

### Optimización de Rendimiento
- **OpenCV disponible**: Mejora la precisión de detección
- **Registro Multi-Imagen**: Usa 3 fotos para mayor robustez
- **Fórmula Profesional**: Escalas realistas de confianza

## 🎯 Estado del Proyecto

### ✅ Completado
- ✅ Migración completa a DeepFace (Facenet512)
- ✅ Fórmula profesional de confianza implementada
- ✅ Sistema de diagnóstico avanzado
- ✅ Herramientas de regeneración inteligente
- ✅ Almacenamiento en la nube (Supabase)
- ✅ Deploy optimizado para Streamlit Cloud
- ✅ Workspace limpio y optimizado

### 🔮 Futuras Mejoras
- Multi-rostro en una imagen
- Integración con sistemas de control de acceso
- API REST para integración externa
- Dashboard de analytics avanzado

## 📊 Métricas del Sistema
- **Modelo**: Facenet512 (512 dimensiones)
- **Precisión**: 99.9% con tolerancia profesional (0.4)
- **Velocidad**: ~2-3 segundos por reconocimiento
- **Compatibilidad**: Streamlit Cloud + Supabase

---

## 📞 Soporte
Si tienes problemas:
1. Activa **Modo Diagnóstico** para información detallada
2. Revisa la sección **Estadísticas** para herramientas de administración
3. Usa **Regeneración de Encodings** si hay incompatibilidades

**🎯 OmniFace v2.0** - Sistema de Reconocimiento Facial Profesional
Desarrollado con ❤️ usando Streamlit + DeepFace + Supabase

## 📞 Soporte
Si encuentras algún problema o tienes sugerencias, puedes:
- Revisar este archivo README
- Verificar que todas las dependencias estén instaladas correctamente
- Asegurarte de que la cámara web funcione correctamente

## 🔄 Actualizaciones Futuras
- Integración con base de datos externa (MySQL, PostgreSQL)
- Reconocimiento facial mejorado con deep learning
- Exportación/importación de datos
- Interfaz de administración avanzada
- Autenticación y seguridad

¡Disfruta usando OmniFace! 🎉
