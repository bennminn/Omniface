# OmniFace - Sistema de Reconocimiento Facial

## 📝 Descripción
OmniFace es una aplicación web desarrollada con Streamlit que permite realizar reconocimiento facial a partir de una base de datos de rostros registrados. La aplicación permite cargar fotografías de personas con sus datos y luego capturar imágenes para identificar a las personas registradas.

## 🚀 Características
- **Gestión de Base de Datos**: Agregar, visualizar y eliminar personas de la base de datos
- **Reconocimiento Facial**: Capturar imágenes con la cámara para identificar personas
- **Interfaz Intuitiva**: Diseño moderno y fácil de usar
- **Almacenamiento Local**: Los datos se guardan en archivos CSV y pickle localmente
- **Estadísticas**: Visualización de métricas del sistema

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- Cámara web (para captura de imágenes)

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Dependencias Incluidas
- streamlit: Framework web para la aplicación
- opencv-python: Procesamiento de imágenes y detección de rostros
- pillow: Manipulación de imágenes
- pandas: Manejo de datos
- numpy: Operaciones matemáticas

## 🏃‍♂️ Ejecución
Para ejecutar la aplicación:

```bash
streamlit run OmnifaceApp.py
```

La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`

## 📱 Uso de la Aplicación

### 1. Gestión de Base de Datos
- **Agregar Personas**: 
  - Ingresa el ID único de la persona
  - Ingresa el nombre completo
  - Sube una fotografía clara del rostro
  - Haz clic en "Agregar Persona"

- **Visualizar Registros**: La tabla muestra todas las personas registradas
- **Eliminar Personas**: Usa los botones de "Eliminar" en cada registro

### 2. Reconocimiento Facial
- Ve a la sección "Reconocimiento Facial"
- Usa la función "Toma una foto" para capturar una imagen
- El sistema analizará automáticamente el rostro detectado
- Se mostrará la información de la persona identificada (si está registrada)

### 3. Estadísticas
- Visualiza métricas del sistema como número de personas registradas
- Revisa el resumen de la base de datos

## 📂 Estructura de Archivos
```
PyWebApps/
├── OmnifaceApp.py          # Aplicación principal
├── requirements.txt        # Dependencias
├── README.md              # Este archivo
├── database.csv           # Base de datos de personas (se crea automáticamente)
├── face_encodings.pkl     # Codificaciones faciales (se crea automáticamente)
└── images/               # Carpeta de imágenes (se crea automáticamente)
    ├── persona1.jpg
    ├── persona2.jpg
    └── ...
```

## 🔧 Configuración Avanzada

### Mejora del Reconocimiento Facial
Para obtener un reconocimiento facial más preciso, puedes instalar la librería `face_recognition`:

#### Opción 1: Instalación con CMake (Recomendado)
1. Instala CMake desde https://cmake.org/download/
2. Asegúrate de agregar CMake al PATH del sistema
3. Ejecuta: `pip install face_recognition`

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
   Esto creará una imagen de prueba que puedes usar

### Error de Cámara
- Verifica que tu cámara web esté conectada y funcionando
- Asegúrate de que no haya otras aplicaciones usando la cámara
- Reinicia el navegador y otorga permisos de cámara

### Error de Dependencias
- Asegúrate de tener todas las dependencias instaladas: `pip install -r requirements.txt`
- Actualiza pip: `python -m pip install --upgrade pip`

### Problemas de Reconocimiento
- La versión actual usa OpenCV para detección básica de rostros
- Para mejor precisión, instala `face_recognition` siguiendo las instrucciones avanzadas

### Mensajes de Error Comunes

**"No se pudo detectar un rostro en la imagen"**
- Usa una imagen con mejor iluminación
- Asegúrate de que el rostro esté centrado y visible
- Prueba con la imagen de ejemplo generada por `create_sample.py`

**"Error guardando en la base de datos"**
- Verifica permisos de escritura en el directorio
- Asegúrate de que no haya archivos bloqueados

**"Error cargando encodings/base de datos"**
- Los archivos pueden estar corruptos
- Elimina `database.csv` y `face_encodings.pkl` para empezar de nuevo

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
