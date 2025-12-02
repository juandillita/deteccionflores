
# Transfer Learning para Clasificación de Flores con MobileNetV2

**TL;DR:** Implementación de un clasificador de flores que reconoce 104 especies diferentes usando Transfer Learning con MobileNetV2 pre-entrenado en ImageNet. Usando TFRecords del dataset Kaggle Flowers con imágenes de 224x224 píxeles, el modelo alcanza alta precisión en tiempo real mediante una aplicación web con TensorFlow.js. La clave del éxito fue congelar las capas base de MobileNetV2 y entrenar solo el clasificador final con Adam optimizer.

## 1. Problema y contexto
- **Tarea:** Clasificación multiclase de flores - reconocer y clasificar 104 especies diferentes de flores a partir de imágenes.
- **¿Por qué Transfer Learning?** Entrenar una CNN desde cero para 104 clases requeriría millones de imágenes etiquetadas y recursos computacionales masivos. MobileNetV2 pre-entrenado en ImageNet ya aprendió características visuales fundamentales (bordes, formas, texturas) que son transferibles a flores, permitiendo obtener alta precisión con menos datos y tiempo de entrenamiento.

## 2. Datos
- **Dataset:** Kaggle Flowers dataset con 104 clases de flores, almacenado en formato TFRecords optimizado para TensorFlow.
- **Estructura:** Imágenes preprocesadas a 224x224 píxeles en formato JPEG, organizadas en TFRecords para carga eficiente.
- **División:** Train/validation split estándar usando el esquema de TFRecords del dataset original.
- **Preprocesamiento:** 
  - Redimensionamiento a 224x224 píxeles (input size de MobileNetV2)
  - Normalización de píxeles a rango [0,1] dividiendo por 255.0
  - Decodificación JPEG con 3 canales RGB
  - Batch size de 32 con prefetch automático para optimización

## 3. Modelo y estrategia de TL
- **Modelo pre-entrenado:** MobileNetV2 de `keras.applications` con pesos de ImageNet (`weights="imagenet"`).
- **Paper/origen:** "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018) - arquitectura optimizada para dispositivos móviles.
- **Estrategia:** Congelamiento completo del backbone (`base.trainable = False`)
  - **Capas congeladas:** Todas las capas de MobileNetV2 pre-entrenado para preservar características visuales aprendidas en ImageNet
  - **Capas entrenables:** Solo la cabeza clasificadora personalizada:
    - GlobalAveragePooling2D para reducir dimensionalidad
    - Dropout(0.2) para regularización
    - Dense(104, activation="softmax") para clasificación multiclase
- **Justificación:** El congelamiento evita catastrophic forgetting y permite entrenamiento rápido con menos riesgo de overfitting.

## 4. Entrenamiento
- **Hiperparámetros:**
  - Batch size: 32
  - Learning rate: 1e-4 (Adam optimizer)
  - Épocas: 20 máximo
  - Loss: sparse_categorical_crossentropy
  - Métricas: accuracy
- **Técnicas de regularización:**
  - **Early stopping:** Paciencia de 3 épocas monitoreando val_accuracy con restore_best_weights=True
  - **Dropout:** 0.2 en la cabeza clasificadora
  - **Data augmentation:** Shuffle del dataset de entrenamiento con buffer de 2048
- **Optimizaciones:**
  - TFRecords con AUTOTUNE para carga paralela eficiente
  - Prefetch automático para pipeline de datos
  - Congelamiento de backbone para entrenar solo ~330K parámetros en lugar de 2.2M

```python
# Arquitectura del modelo - TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Cargar MobileNetV2 pre-entrenado sin la cabeza clasificadora
base = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # Congelar todas las capas del backbone

# Construir modelo con cabeza personalizada
inputs = keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)  # Base congelada
x = layers.GlobalAveragePooling2D()(x)  # Pooling global
x = layers.Dropout(0.2)(x)  # Regularización
outputs = layers.Dense(104, activation="softmax")(x)  # 104 clases de flores

model = keras.Model(inputs, outputs)

# Compilar con Adam y sparse categorical crossentropy
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## 5. Resultados y evaluación
- **Métrica principal:** Accuracy en validation set durante el entrenamiento
- **Convergencia:** Early stopping activado automáticamente cuando val_accuracy deja de mejorar
- **Tiempo de entrenamiento:** Significativamente reducido gracias al congelamiento del backbone
- **Modelo final:** Exportado automáticamente a TensorFlow.js para despliegue web en tiempo real
- **Aplicación práctica:** 
  - Interfaz web responsive que funciona en móviles y desktop
  - Predicciones en tiempo real desde la cámara del dispositivo
  - Top-5 predicciones mostradas con barras de confianza
  - 104 especies de flores correctamente etiquetadas con nombres reales
- **Ventajas del Transfer Learning:**
  - Sin TL: Requeriría entrenar 2.2M+ parámetros desde cero
  - Con TL: Solo entrena ~330K parámetros de la cabeza clasificadora
  - Convergencia más rápida y estable gracias a features pre-aprendidas

## 6. Lecciones y límites
- **¿Qué funcionó bien?**
  - MobileNetV2 demostró ser ideal para deployment móvil - modelo ligero pero efectivo
  - El congelamiento completo del backbone previno overfitting eficazmente
  - TensorFlow.js permite deployment instantáneo sin necesidad de servidor
  - Early stopping automático evitó sobreentrenamiento
  
- **Limitaciones identificadas:**
  - **Dependiente de iluminación:** Funciona mejor con buena iluminación natural
  - **Ángulo y distancia:** Requiere flores bien enfocadas y centradas en la imagen
  - **Solapamiento de especies:** Algunas flores muy similares pueden confundirse
  - **Sesgo del dataset:** Limitado a las 104 especies incluidas en el dataset original
  
- **Consideraciones éticas:**
  - Dataset público y licencia permisiva
  - No hay datos personales o sensibles involucrados
  - Uso educativo y científico para identificación botánica
  - Limitaciones claras: no reemplaza identificación experta para usos críticos

## 7. Reproducibilidad
- **Repositorio**: [GitHub del proyecto FlowerDetection]
- **Demo en vivo**: Aplicación web disponible ejecutando `python -m http.server 8000` en el directorio del proyecto
- **Archivos clave**:
  - `train_flowers_tl.py`: Script de entrenamiento con Transfer Learning
  - `flower_detection.ipynb`: Notebook con proceso completo paso a paso
  - `index.html`: Aplicación web con interfaz móvil optimizada
  - `requirements.txt`: Dependencias exactas (TensorFlow 2.13.0, TensorFlow.js 4.5.0)
- **Modelo pre-entrenado**: Disponible en `web_model_flowers/` listo para usar
- **Datos**: Dataset Kaggle Flowers en formato TFRecords (no incluido, se descarga automáticamente)
- **Seed**: Configurada en el script de entrenamiento para resultados reproducibles

## 8. Referencias
- **MobileNetV2 Paper**: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." IEEE/CVF Conference on Computer Vision and Pattern Recognition.
- **Dataset**: Kaggle Flowers Recognition Dataset - 104 flower categories
- **TensorFlow**: TensorFlow 2.13.0 documentation y Keras API
- **TensorFlow.js**: TensorFlow.js 4.5.0 para deployment web y móvil
- **Transfer Learning**: Pan, S. J., & Yang, Q. (2009). "A survey on transfer learning." IEEE Transactions on knowledge and data engineering.
- **Bootstrap 5**: Framework CSS usado para la interfaz responsive
- **Inspiración**: Proyectos similares de clasificación de imágenes con Transfer Learning en la comunidad TensorFlow


---
## ✅ Checklist de Publicación (Transfer Learning)

**Antes de publicar, verifica:**

1. **Reproducibilidad**
   - [ ] Notebook/Colab enlazado y ejecuta de inicio a fin sin errores.
   - [ ] Seeds fijadas (`random`, `numpy`, `torch/tf`) y aviso del runtime (CPU/GPU, T4/V100/A100).
   - [ ] `requirements.txt` / `environment.yml` o celda `pip install ...` clara.
   - [ ] Script/notebook descarga y organiza datos (o explica cómo obtenerlos).

2. **Exactitud técnica**
   - [ ] Indicas modelo pre-entrenado (paper, checkpoint) y qué capas congelas/descongelas.
   - [ ] Justificas hiperparámetros clave (lr, batch, epochs, scheduler/early stopping).
   - [ ] Métricas correctas para la tarea (accuracy/F1/AUC/BLEU/mAP) y sus curvas/tablas.
   - [ ] Incluyes *sanity checks* (formas, batch pequeño, overfit a pocas muestras).

3. **Ética y licencias**
   - [ ] Citas dataset y modelo con enlaces y **licencias**.
   - [ ] Declaras posibles **sesgos** y límites del modelo.
   - [ ] Código con licencia clara (MIT/Apache-2.0/BSD-3-Clause o equivalente).

4. **Presentación**
   - [ ] TL;DR inicial (3–4 líneas) y conclusiones al final (3–5 bullets).
   - [ ] Imágenes con texto alternativo y gráficos legibles (títulos/leyendas).
   - [ ] Ortografía y formato (títulos, listas, enlaces verificados).
   - [ ] Enlaces a repo/weights/demo (si aplica) funcionando.

5. **Seguridad y privacidad**
   - [ ] No publicas claves/tokens/credenciales en el código o logs.
   - [ ] Si hay datos sensibles, usas versiones anonimizadas o *mock data*.

**Entrega:** URL del artículo + URL del repo + URL del Colab + vídeo donde estás usando el celular para el reconocimiento
