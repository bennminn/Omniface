"""
Módulo para manejar la base de datos Supabase
"""
import streamlit as st
import numpy as np
import pandas as pd
from supabase import create_client, Client
import pickle
import base64
import os
from typing import Dict, Optional, Tuple, Any
import io
from PIL import Image

class SupabaseManager:
    def __init__(self):
        """Inicializar conexión con Supabase"""
        self.supabase = self._get_supabase_client()
        self._ensure_tables_exist()
    
    def _get_supabase_client(self) -> Client:
        """Crear cliente de Supabase usando secrets o variables de entorno"""
        try:
            # Intentar usar Streamlit secrets (para deploy)
            if hasattr(st, 'secrets') and 'supabase' in st.secrets:
                url = st.secrets["supabase"]["url"]
                key = st.secrets["supabase"]["key"]
            else:
                # Usar variables de entorno para desarrollo local
                from dotenv import load_dotenv
                load_dotenv()
                url = os.getenv("SUPABASE_URL")
                key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                st.error("❌ Configuración de Supabase no encontrada")
                st.info("💡 Por favor configura las credenciales de Supabase")
                st.stop()
            
            return create_client(url, key)
        except Exception as e:
            st.error(f"❌ Error conectando a Supabase: {e}")
            st.stop()
    
    def _ensure_tables_exist(self):
        """Asegurar que las tablas existan en Supabase"""
        try:
            # Verificar si las tablas existen, si no, mostrar instrucciones
            response = self.supabase.table('personas').select('*').limit(1).execute()
        except Exception as e:
            st.error("❌ Las tablas de la base de datos no existen")
            st.info("""
            📋 **Instrucciones para configurar Supabase:**
            
            1. Ve a https://supabase.com y crea una cuenta gratuita
            2. Crea un nuevo proyecto
            3. Ve a SQL Editor y ejecuta estas consultas:
            
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
            
            -- Política para permitir todas las operaciones (puedes hacer esto más restrictivo)
            CREATE POLICY "Permitir todas las operaciones" ON personas
                FOR ALL USING (true);
            ```
            
            4. Copia tu URL y Anon Key del proyecto
            5. Configúralos en los secrets de Streamlit o en .env
            """)
            st.stop()
    
    def save_person(self, person_id: str, name: str, image: Image.Image, encoding: np.ndarray) -> bool:
        """Guardar una persona en la base de datos"""
        try:
            # Convertir imagen a base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Convertir encoding a string base64
            encoding_bytes = pickle.dumps(encoding)
            encoding_base64 = base64.b64encode(encoding_bytes).decode()
            
            # Insertar en Supabase
            response = self.supabase.table('personas').insert({
                'id': person_id,
                'nombre': name,
                'imagen_base64': img_base64,
                'encoding_data': encoding_base64
            }).execute()
            
            return len(response.data) > 0
        except Exception as e:
            st.error(f"Error guardando persona: {e}")
            return False
    
    def get_all_persons(self) -> pd.DataFrame:
        """Obtener todas las personas de la base de datos"""
        try:
            response = self.supabase.table('personas').select('id, nombre, fecha_creacion').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                # Añadir columna imagen_path para compatibilidad
                df['imagen_path'] = df['id'].apply(lambda x: f"supabase_image_{x}")
                return df
            else:
                return pd.DataFrame(columns=['id', 'nombre', 'imagen_path', 'fecha_creacion'])
        except Exception as e:
            st.error(f"Error cargando personas: {e}")
            return pd.DataFrame(columns=['id', 'nombre', 'imagen_path', 'fecha_creacion'])
    
    def get_person_image(self, person_id: str) -> Optional[Image.Image]:
        """Obtener imagen de una persona"""
        try:
            response = self.supabase.table('personas').select('imagen_base64').eq('id', person_id).execute()
            
            if response.data and response.data[0]['imagen_base64']:
                img_data = base64.b64decode(response.data[0]['imagen_base64'])
                return Image.open(io.BytesIO(img_data))
            return None
        except Exception as e:
            st.error(f"Error cargando imagen: {e}")
            return None
    
    def get_all_encodings(self) -> Dict[str, np.ndarray]:
        """Obtener todas las codificaciones faciales"""
        try:
            response = self.supabase.table('personas').select('id, encoding_data').execute()
            
            encodings = {}
            for person in response.data:
                if person['encoding_data']:
                    encoding_bytes = base64.b64decode(person['encoding_data'])
                    encoding = pickle.loads(encoding_bytes)
                    encodings[person['id']] = encoding
            
            return encodings
        except Exception as e:
            st.error(f"Error cargando encodings: {e}")
            return {}
    
    def delete_person(self, person_id: str) -> bool:
        """Eliminar una persona de la base de datos"""
        try:
            response = self.supabase.table('personas').delete().eq('id', person_id).execute()
            return len(response.data) > 0
        except Exception as e:
            st.error(f"Error eliminando persona: {e}")
            return False
    
    def person_exists(self, person_id: str) -> bool:
        """Verificar si una persona existe"""
        try:
            response = self.supabase.table('personas').select('id').eq('id', person_id).execute()
            return len(response.data) > 0
        except Exception as e:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de datos"""
        try:
            response = self.supabase.table('personas').select('*').execute()
            total_persons = len(response.data)
            
            # Calcular tamaño total aproximado
            total_size = 0
            for person in response.data:
                if person.get('imagen_base64'):
                    total_size += len(person['imagen_base64'])
                if person.get('encoding_data'):
                    total_size += len(person['encoding_data'])
            
            return {
                'total_persons': total_persons,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'encodings_active': sum(1 for p in response.data if p.get('encoding_data'))
            }
        except Exception as e:
            st.error(f"Error obteniendo estadísticas: {e}")
            return {'total_persons': 0, 'total_size_mb': 0, 'encodings_active': 0}

# Instancia global del manager
@st.cache_resource
def get_db_manager():
    """Obtener instancia singleton del manager de base de datos"""
    return SupabaseManager()
