#!/usr/bin/env python3
"""
Crear una imagen de ejemplo con un rostro simulado para probar OmniFace
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

def create_sample_face():
    """Crear una imagen de ejemplo con un rostro básico"""
    # Crear una imagen de 300x300 píxeles
    img = Image.new('RGB', (300, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Dibujar un rostro básico
    # Cara (círculo)
    draw.ellipse([75, 75, 225, 225], fill='peachpuff', outline='black', width=2)
    
    # Ojos
    draw.ellipse([100, 120, 130, 140], fill='white', outline='black', width=1)
    draw.ellipse([170, 120, 200, 140], fill='white', outline='black', width=1)
    
    # Pupilas
    draw.ellipse([110, 125, 120, 135], fill='black')
    draw.ellipse([180, 125, 190, 135], fill='black')
    
    # Nariz
    draw.ellipse([145, 155, 155, 170], fill='rosybrown', outline='black', width=1)
    
    # Boca
    draw.arc([130, 180, 170, 200], start=0, end=180, fill='red', width=3)
    
    # Cabello
    draw.ellipse([85, 85, 215, 130], fill='brown', outline='black', width=1)
    
    return img

def main():
    """Crear imagen de ejemplo"""
    print("🎨 Creando imagen de ejemplo para OmniFace...")
    
    # Crear directorio si no existe
    os.makedirs("sample_images", exist_ok=True)
    
    # Crear la imagen
    sample_face = create_sample_face()
    
    # Guardar la imagen
    sample_path = "sample_images/persona_ejemplo.jpg"
    sample_face.save(sample_path)
    
    print(f"✅ Imagen de ejemplo creada: {sample_path}")
    print("📝 Puedes usar esta imagen para probar la funcionalidad de agregar persona")
    print("🔧 Datos sugeridos:")
    print("   - ID: EJEMPLO001")
    print("   - Nombre: Persona de Ejemplo")
    print(f"   - Imagen: {sample_path}")

if __name__ == "__main__":
    main()
