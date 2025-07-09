# OmniFace v2.0 - Sistema de Reconocimiento Facial con Supabase

Sistema avanzado de reconocimiento facial con integración a Supabase para persistencia de datos en la nube.

## 🌟 Características

- ✨ **Reconocimiento facial en tiempo real**
- 📸 **Captura con cámara web integrada**
- 📁 **Subida de archivos de imagen**
- ☁️ **Almacenamiento en la nube con Supabase**
- 🎯 **Interfaz intuitiva con Streamlit**
- 🔒 **Datos persistentes**

## 🚀 Deploy en Streamlit Cloud

### Configuración de Supabase

1. Ve a [supabase.com](https://supabase.com) y crea una cuenta
2. Crea un nuevo proyecto
3. Ve a SQL Editor y ejecuta:

```sql
-- Tabla para personas
CREATE TABLE personas (
    id TEXT PRIMARY KEY,
    nombre TEXT NOT NULL,
    imagen_base64 TEXT,
    encoding_data TEXT,
    fecha_creacion TIMESTAMP DEFAULT NOW()
);

-- Habilitar RLS (Row Level Security)
ALTER TABLE personas ENABLE ROW LEVEL SECURITY;

-- Política para permitir todas las operaciones
CREATE POLICY "Permitir todas las operaciones" ON personas
    FOR ALL USING (true);
```

4. Copia tu URL del proyecto y Anon Key

## 🛠️ Desarrollo Local

### Requisitos

```bash
pip install -r requirements.txt
```

### Configuración Local

Tienes dos opciones para configurar las credenciales:

**Opción 1: Usando .env (recomendado para variables de entorno)**
1. Copia `.env.example` a `.env`
2. Completa las credenciales de Supabase

**Opción 2: Usando secrets.toml (nativo de Streamlit)**
1. Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml`
2. Completa las credenciales:

```toml
[supabase]
url = "tu-url-de-supabase"
key = "tu-anon-key"
```

### Ejecutar la aplicación

```bash
streamlit run OmnifaceApp.py
```

### Configuración en Streamlit Cloud

1. Sube este repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. En "Advanced settings", agrega estos secrets:

```toml
[supabase]
url = "tu_supabase_url_aqui"
key = "tu_supabase_anon_key_aqui"
```

5. ¡Deploy automático!

##  Características Técnicas

- **Base de datos:** Supabase (PostgreSQL)
- **Almacenamiento:** Imágenes en Base64
- **Encodings:** Vectores normalizados con OpenCV
- **Comparación:** Distancia coseno
- **Interfaz:** Streamlit con tabs dinámicas

## 🎯 Funcionalidades

### Gestión de Personas
- Agregar personas con foto o cámara
- Galería visual de rostros registrados
- Eliminación segura de registros

### Reconocimiento
- Análisis en tiempo real
- Porcentaje de confianza
- Imagen de referencia comparativa

### Estadísticas
- Contador de personas registradas
- Métricas de almacenamiento
- Estado del sistema

---

**Desarrollado con ❤️ usando Streamlit + Supabase**
