"""
Módulo para manejo robusto de DeepFace con fallback
"""
import numpy as np

# Configurar TensorFlow para reducir warnings
try:
    from tf_config import *
except ImportError:
    # Configuración básica si tf_config no está disponible
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Variables globales para estado
DEEPFACE_AVAILABLE = False
TENSORFLOW_VERSION = None
DeepFace = None

def initialize_deepface():
    """Inicializar DeepFace con manejo robusto de errores"""
    global DEEPFACE_AVAILABLE, TENSORFLOW_VERSION, DeepFace
    
    try:
        # Verificar TensorFlow primero
        import tensorflow as tf
        TENSORFLOW_VERSION = tf.__version__
        
        # Parche específico para LocallyConnected2D en diferentes entornos
        try:
            # Intentar parche previo al import de DeepFace
            import sys
            import tensorflow.keras.layers as keras_layers
            
            # Si LocallyConnected2D no está disponible, crear un parche
            if not hasattr(keras_layers, 'LocallyConnected2D'):
                # Importar desde ubicación alternativa si existe
                try:
                    import keras.layers
                    if hasattr(keras.layers, 'LocallyConnected2D'):
                        keras_layers.LocallyConnected2D = keras.layers.LocallyConnected2D
                except ImportError:
                    pass
                    
        except Exception as e:
            pass  # Continuar sin parche si falla
        
        # Intentar importar DeepFace
        from deepface import DeepFace as DF
        DeepFace = DF
        DEEPFACE_AVAILABLE = True
        
        return True, f"DeepFace inicializado correctamente (TF: {TENSORFLOW_VERSION})"
        
    except ImportError as e:
        error_msg = str(e).lower()
        
        if "locallyconnected2d" in error_msg:
            return False, f"Modo simulado activo (compatibilidad TensorFlow)"
        elif "tensorflow" in error_msg or "keras" in error_msg:
            return False, f"Modo simulado activo (compatibilidad de dependencias)"
        else:
            return False, f"Modo simulado activo (DeepFace no disponible)"
            
    except Exception as e:
        return False, f"Error inesperado. Modo simulado activado."

def get_deepface_fallback():
    """Crear implementación de fallback para DeepFace"""
    class DeepFaceFallback:
        @staticmethod
        def represent(img_path, model_name='Facenet512', enforce_detection=True, **kwargs):
            """Simular generación de encoding de 512 dimensiones"""
            # Generar embedding determinista basado en el path para consistencia
            if isinstance(img_path, str):
                # Usar hash del path para consistencia
                np.random.seed(hash(img_path) % 2**32)
            
            embedding = np.random.rand(512).tolist()
            return [{"embedding": embedding}]
        
        @staticmethod
        def verify(img1_path, img2_path, model_name='Facenet512', enforce_detection=True, **kwargs):
            """Simular verificación facial"""
            # Simular distancia basada en paths para cierta consistencia
            if isinstance(img1_path, str) and isinstance(img2_path, str):
                # Misma imagen = menor distancia
                if img1_path == img2_path:
                    distance = np.random.uniform(0.1, 0.3)
                else:
                    distance = np.random.uniform(0.4, 0.8)
            else:
                distance = np.random.uniform(0.2, 0.8)
            
            return {
                "verified": distance < 0.5,
                "distance": distance,
                "model": model_name
            }
    
    return DeepFaceFallback()

def get_deepface_instance():
    """Obtener instancia de DeepFace (real o fallback)"""
    if DEEPFACE_AVAILABLE and DeepFace is not None:
        return DeepFace
    else:
        return get_deepface_fallback()

def is_deepface_available():
    """Verificar si DeepFace está disponible"""
    return DEEPFACE_AVAILABLE

def get_tensorflow_version():
    """Obtener versión de TensorFlow si está disponible"""
    return TENSORFLOW_VERSION
