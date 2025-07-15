# Configuración para reducir warnings de TensorFlow
import os
import warnings

# Suprimir warnings específicos de TensorFlow que no son críticos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo errores
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deshabilitar oneDNN para consistencia

# Suprimir warnings específicos de compatibilidad
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*tf.reset_default_graph.*')
warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
