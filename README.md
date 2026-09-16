# 🛰️ Estigia CubeSat Chatbot - Pluton UPV

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Edge_AI-white.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Telemetry-orange.svg)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry_Pi-c51a4a.svg)

Un asistente conversacional ultraligero y multilingüe diseñado para ejecutarse localmente en una **Raspberry Pi** sin conexión a internet. Este proyecto da voz a **Estigia**, un satélite (CubeSat) del equipo Pluton UPV, combinando la potencia de Modelos de Lenguaje Grandes (LLMs) cuantizados con un clasificador de Machine Learning ultrarrápido para telemetría.

## ✨ Características Principales

* 🌍 **Trilingüe Nativo:** Soporta Español, Valencià e Inglés mediante *System Prompts* dinámicos que fuerzan al modelo a responder en el idioma seleccionado sin sobrecargar la memoria.
* ⚡ **Rutas de Ejecución Duales:**
    * **Ruta Rápida (Telemetría):** Utiliza un modelo `LinearSVC` (vía `joblib` y `scikit-learn`) para detectar intenciones de sensores (Altitud, Temperatura) y responder en milisegundos.
    * **Ruta Cognitiva (LLM):** Utiliza [Ollama](https://ollama.com/) para mantener conversaciones abiertas sobre ciencia, el espacio y la misión.
* 📊 **Métricas en Tiempo Real:** Monitorización integrada de latencia, *Time To First Token* (TTFT) y velocidad de generación (Tokens por segundo), ideal para detectar sobrecalentamiento (*thermal throttling*) en la Raspberry Pi.
* 🧠 **Gestión de RAM:** Historial de conversación circular (FIFO) que evita bloqueos por falta de memoria (OOM) en dispositivos con recursos limitados.
* 🌊 **Streaming de Texto:** La interfaz de terminal imprime la respuesta token a token, eliminando la sensación de espera.

---

## 🛠️ Requisitos del Sistema

* **Hardware:** Raspberry Pi 4 (8GB RAM) o Raspberry Pi 5 recomendada. (También funciona en PC/Mac).
* **Software:** Linux / macOS / Windows con Python 3.9 o superior.
* **Motor de IA:** [Ollama](https://ollama.com/) instalado y ejecutándose en segundo plano.

## 📦 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/franciscobdl/EstigiaChatbot.git
   cd EstigiaChatbot
   ```

2. **Crear y activar un entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   # venv\Scripts\activate   # En Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: La versión optimizada para Raspberry Pi (`ollama_launch_2_1.py`) no requiere cargar pesados modelos de traducción en memoria.*

4. **Descargar el modelo en Ollama:**

   4.1. **Instalar Ollama**:

    Linux / Raspberry Pi: 
        
        curl -fsSL https://ollama.com/install.sh | sh

    Windows / macOS: Descarga el instalador desde [ollama.com](https://ollama.com/).

   4.2. **Descargar el modelo de Estigia**:
   
   Una vez instalado Ollama, descarga el modelo optimizado para este proyecto ejecutando:

   ```bash
   ollama pull franciscobdl/Estigia2
   ```

---

## 🚀 Uso

Inicia el sistema ejecutando el script principal:

```bash
python ollama_launch_2_1.py
```

### Comandos de la Interfaz
Durante la ejecución, el usuario puede usar los siguientes comandos especiales:
* `/lang` : Reinicia el historial y vuelve al menú de selección de idioma.
* `/stop` o `exit` : Apaga el sistema de forma segura.

### Ejemplo de Ejecución
```text
--- STARTING ESTIGIA SYSTEMS ON RASPBERRY PI ---
⚙️ Loading telemetry classifier (joblib)...
✅ Telemetry loaded in 0.05 seconds.
🧠 Waking up model 'franciscobdl/Estigia2' in Ollama...
✅ Model loaded and ready in 1.20 seconds.

Select communication language:
1. Español  🇪🇸
2. Valencià 🦇
3. English  🇬🇧
Option (1/2/3) or '/stop' to quit: 1

✅ Idioma configurado: Español
Escribe '/lang' para cambiar de idioma, o '/stop' para salir.
----------------------------------------

👤 Usuario: ¿Qué temperatura hace ahí fuera?
🛰️ Estigia: 📡 Temperatura interna: 22.4 ºC
[⏱️ SENSOR | Tiempo total: 0.0034s | Modo: API Rápida]

👤 Usuario: ¿Qué opinas de los humanos?
🛰️ Estigia: ¡Me caen genial! Sois como pequeños exploradores anclados a la Tierra, pero con mentes que viajan más rápido que la luz. 
[⏱️ LLM | TTFT: 1.12s | Velocidad: 9.85 t/s | Tokens: 32]
```

---

## 📁 Estructura del Proyecto

```text
EstigiaChatbot/
├── ollama_launch_2_1.py         # Script principal optimizado para Edge AI
├── requirements.txt             # Dependencias de Python
├── Models/
│   └── telemetry_classifier.joblib  # Modelo entrenado para detectar intenciones
│   └── modelo-PLUTON_UPV_es_svm.joblib  # Modelo entrenado para clasificar preguntas de telemetría en español
│   └── modelo-PLUTON_UPV_ca_svm.joblib  # Modelo entrenado para clasificar preguntas de telemetría en catalán
│   └── modelo-PLUTON_UPV_en_svm.joblib  # Modelo entrenado para clasificar preguntas de telemetría en inglés
└── README.md                    # Documentación del proyecto
```

## 🧠 Personalidad del Modelo
El *System Prompt* inyectado fuerza a la IA a adoptar el rol de **Estigia**: un satélite joven, curioso, alegre y fascinado por el conocimiento, asegurando respuestas seguras (sin sarcasmo ni contenido adulto) y repletas de metáforas espaciales.

## 🤝 Créditos
Desarrollado para el equipo aeroespacial **Pluton UPV** de la Universitat Politècnica de València.