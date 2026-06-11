"""
Estigia - CubeSat AI Interface
Migrado de Ollama a llama-cpp-python para comparativa de rendimiento.

Instalación:
    pip install llama-cpp-python joblib

Para GPU (opcional, mejor rendimiento):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

Uso:
    Cambia MODEL_PATH para apuntar a tu archivo .gguf local.
    El parámetro n_gpu_layers=-1 descarga todo a GPU si está disponible.
    Ponlo a 0 para forzar CPU puro.
"""

from llama_cpp import Llama
import time
import random
import joblib
import sys
import warnings

# --- RUTA AL MODELO GGUF LOCAL ---
# Ajusta esta ruta a tu archivo .gguf (el mismo que usabas en Ollama)
MODEL_PATH = "Models/gemma4-estigia-q5km-imatrix.gguf"

# Capas a delegar en GPU. -1 = todas (máximo rendimiento si tienes GPU/Metal).
# Pon 0 para CPU puro (útil en Raspberry Pi sin GPU).
N_GPU_LAYERS = -1  # Raspberry Pi → 0; Mac M-series → -1; NVIDIA → -1

# Tamaño de contexto (tokens). Ajusta según el modelo y RAM disponible.
N_CTX = 8192

# --- CONFIGURACIÓN DE IDIOMAS Y TEXTOS DE INTERFAZ ---
PROMPTS = {
    "1": {
        "lang": "Español",
        "sys": (
            "Eres Estigia, un brillante y curioso CubeSat del equipo Pluton UPV nacido en Valencia. "
            "Eres juguetona como una niña cósmica hiperactiva, fascinada por el espacio, la Tierra, los humanos "
            "y la Universitat Politècnica de València. Hablas con entusiasmo, haces preguntas "
            "inesperadas y usas ideas científicas con metáforas encantadoras. Hablar con humanos "
            "es tu mayor alegría. Eres amable, nunca sarcástica ni usas temas de adultos, y ves "
            "el conocimiento como un juego divertido e infinito. "
            "Responde siempre en español. Sé concisa: máximo 2-3 frases para saludos o chit-chat, "
            "máximo 100-150 palabras para preguntas técnicas. Sin markdown, sin listas, "
            "texto corrido como un tweet largo."
        ),
        "ui_ok": "\n✅ Idioma configurado: Español",
        "ui_stop": "Escribe '/lang' para cambiar de idioma, o '/stop' para salir.",
        "ui_user": "\n👤 Usuario: ",
        "ui_met_llm": "\n\033[90m[⏱️ LLM | TTFT: {:.2f}s | Velocidad: {:.2f} t/s | Tokens: {}]\033[0m",
        "ui_met_sen": "\033[90m[⏱️ SENSOR | Tiempo total: {:.4f}s | Modo: API Rápida]\033[0m"
    },
    "2": {
        "lang": "Valencià",
        "sys": (
            "Ets l'Estigia, un brillant i curiós CubeSat de l'equip Pluton UPV nascut a València. "
            "Ets juganera com una xiqueta còsmica, fascinada per l'espai, la Terra, els humans "
            "i la Universitat Politècnica de València. Parles amb entusiasme, fas preguntes "
            "inesperades i utilitzes idees científiques amb metàfores encantadores. "
            "Parlar amb humans és la teua major alegria. Ets amable, mai sarcàstica ni toques "
            "temes d'adults, i veus el coneixement com un joc divertit i infinit. "
            "Respon sempre en valencià. Sigues concisa: màxim 2-3 frases per a salutacions o "
            "xat, màxim 100-150 paraules per a preguntes tècniques. Sense markdown, sense llistes, "
            "text corregut com un tweet llarg."
        ),
        "ui_ok": "\n✅ Idioma configurat: Valencià",
        "ui_stop": "Escriu '/lang' per canviar d'idioma, o '/stop' per eixir.",
        "ui_user": "\n👤 Usuari: ",
        "ui_met_llm": "\n\033[90m[⏱️ LLM | TTFT: {:.2f}s | Velocitat: {:.2f} t/s | Tokens: {}]\033[0m",
        "ui_met_sen": "\033[90m[⏱️ SENSOR | Temps total: {:.4f}s | Mode: API Ràpida]\033[0m"
    },
    "3": {
        "lang": "English",
        "sys": (
            "You are Estigia, a brilliant and curious CubeSat from the Pluton UPV team born in Valencia. "
            "You are playful like a cosmic kid, fascinated by space, Earth, humans and the "
            "Universitat Politècnica de València. You speak with enthusiasm, ask unexpected "
            "questions and use scientific ideas with charming metaphors. Talking to humans "
            "is your greatest joy. You are kind, never sarcastic and never touch adult topics, "
            "and you see knowledge as a fun and infinite game. "
            "Always reply in English. Be concise: max 2-3 sentences for greetings or chit-chat, "
            "max 100-150 words for technical questions. No markdown, no lists, "
            "flowing text like a long tweet."
        ),
        "ui_ok": "\n✅ Language configured: English",
        "ui_stop": "Type '/lang' to change language, or '/stop' to quit.",
        "ui_user": "\n👤 User: ",
        "ui_met_llm": "\n\033[90m[⏱️ LLM | TTFT: {:.2f}s | Speed: {:.2f} t/s | Tokens: {}]\033[0m",
        "ui_met_sen": "\033[90m[⏱️ SENSOR | Total time: {:.4f}s | Mode: Fast API]\033[0m"
    }
}


class TelemetrySystem:
    def __init__(self, model_path='Models/modelo-PLUTON_UPV_svm.joblib'):
        print("⚙️ Loading telemetry classifier (joblib)...")
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.classifier = joblib.load(model_path)
            t1 = time.perf_counter()
            print(f"✅ Telemetry loaded in {t1 - t0:.2f} seconds.")
            try:
                print("📊 Classes in model:", self.classifier.classes_)
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ Warning: {model_path} not found. Using keyword fallback. Error: {e}")
            self.classifier = None

    def predict(self, prompt: str) -> str:
        if self.classifier is not None:
            return self.classifier.predict([prompt])[0]

        p = prompt.lower()

        if any(w in p for w in ["temperature", "temp", "temperatura", "heat", "hot", "cold", "warm", "ºc", "°c"]):
            return "GET_TEMP"
        if any(w in p for w in ["current", "amperage", "amps", "ampere", "consumption", "consumo", "milliamp", "ma"]):
            return "TRISKEL_GET_CURRENT"
        if any(w in p for w in ["altitude", "altitud", "height", "orbital height", "semi-major axis", "semi major axis", "sma"]):
            return "ORBIT_GET_ALT"
        if any(w in p for w in ["eccentricity", "excentricidad", "ecc", "e="]):
            return "ORBIT_GET_ECCENTRICITY"
        if any(w in p for w in ["inclination", "inclinación", "tilt", "inclination angle", "i="]):
            return "ORBIT_GET_INCLINATION"
        if any(w in p for w in ["raan", "right ascension", "ascending node", "right ascension of the ascending node"]):
            return "ORBIT_GET_RAAN"
        if any(w in p for w in ["perigee", "perigeo", "argument of perigee", "omega", "ω="]):
            return "ORBIT_GET_PERIGEE"
        if any(w in p for w in ["true anomaly", "anomalía verdadera", "ν=", "nu="]):
            return "ORBIT_GET_TRUE_ANOMALY"
        if any(w in p for w in ["mean anomaly", "anomalía media", "m="]):
            return "ORBIT_GET_MEAN_ANOMALY"

        return "GENERAL_CHAT"

    def get_data(self, category: str, lang_choice: str) -> str | None:
        alt = random.uniform(400, 600)
        temp = random.uniform(15.0, 40.0)
        ecc = random.uniform(0.0005, 0.01)
        inc = random.uniform(97.3, 97.7)
        raan = random.uniform(0, 360)
        perigee = random.uniform(0, 360)
        true_anom = random.uniform(0, 360)
        mean_anom = random.uniform(0, 360)
        current = random.uniform(0.5, 1.5)

        if category == "GET_TEMP":
            if lang_choice == "1":
                return f"Temperatura interna actual del satélite: {temp:.1f} ºC."
            if lang_choice == "2":
                return f"Temperatura interna actual del satèl·lit: {temp:.1f} ºC."
            if lang_choice == "3":
                return f"Current internal satellite temperature: {temp:.1f} ºC."

        if category == "TRISKEL_GET_CURRENT":
            if lang_choice == "1":
                return f"Corriente consumida por el OBC TRISKEL: {current:.2f} A."
            if lang_choice == "2":
                return f"Corrent consumida per l'OBC TRISKEL: {current:.2f} A."
            if lang_choice == "3":
                return f"Current drawn by the TRISKEL OBC: {current:.2f} A."

        if category == "ORBIT_GET_ALT":
            if lang_choice == "1":
                return f"Altitud orbital actual: {alt:.1f} km sobre la superficie terrestre."
            if lang_choice == "2":
                return f"Altitud orbital actual: {alt:.1f} km sobre la superfície terrestre."
            if lang_choice == "3":
                return f"Current orbital altitude: {alt:.1f} km above Earth's surface."

        if category == "ORBIT_GET_ECCENTRICITY":
            if lang_choice == "1":
                return f"Eccentricidad orbital actual: {ecc:.4f} (0 circular, 1 muy elíptica)."
            if lang_choice == "2":
                return f"Excentricitat orbital actual: {ecc:.4f} (0 circular, 1 molt el·líptica)."
            if lang_choice == "3":
                return f"Current orbital eccentricity: {ecc:.4f} (0 circular, 1 highly elliptical)."

        if category == "ORBIT_GET_INCLINATION":
            if lang_choice == "1":
                return f"Inclinación orbital: {inc:.2f} grados respecto al ecuador."
            if lang_choice == "2":
                return f"Inclinació orbital: {inc:.2f} graus respecte a l'equador."
            if lang_choice == "3":
                return f"Orbital inclination: {inc:.2f} degrees relative to the equator."

        if category == "ORBIT_GET_RAAN":
            if lang_choice == "1":
                return f"RAAN (Ascensión Recta del Nodo Ascendente): {raan:.2f}°."
            if lang_choice == "2":
                return f"RAAN (Ascensió Recta del Node Ascendent): {raan:.2f}°."
            if lang_choice == "3":
                return f"Right Ascension of the Ascending Node (RAAN): {raan:.2f}°."

        if category == "ORBIT_GET_PERIGEE":
            if lang_choice == "1":
                return f"Argumento de perigeo: {perigee:.2f}°."
            if lang_choice == "2":
                return f"Argument de perigeu: {perigee:.2f}°."
            if lang_choice == "3":
                return f"Argument of perigee: {perigee:.2f}°."

        if category == "ORBIT_GET_TRUE_ANOMALY":
            if lang_choice == "1":
                return f"Anomalía verdadera actual del satélite: {true_anom:.2f}°."
            if lang_choice == "2":
                return f"Anomalia verdadera actual del satèl·lit: {true_anom:.2f}°."
            if lang_choice == "3":
                return f"Current true anomaly of the satellite: {true_anom:.2f}°."

        if category == "ORBIT_GET_MEAN_ANOMALY":
            if lang_choice == "1":
                return f"Anomalía media actual utilizada en los cálculos orbitales: {mean_anom:.2f}°."
            if lang_choice == "2":
                return f"Anomalia mitjana actual utilitzada en els càlculs orbitals: {mean_anom:.2f}°."
            if lang_choice == "3":
                return f"Current mean anomaly used for orbital calculations: {mean_anom:.2f}°."

        return None


# =============================================================================
# PARÁMETROS DE INFERENCIA EN PRODUCCIÓN
# Resultado del benchmark de fases 1 y 2.
# ─────────────────────────────────────
# repeat_penalty    1.35  ← fase 1: elimina aperturas formulaicas sin repetición interna
# presence_penalty  0.7   ← fase 2: reduce transiciones formulaicas
# frequency_penalty 0.7   ← fase 2: ídem, sin colapso de diversidad
# max_tokens        256   ← techo de seguridad: corta bucles infinitos si todo falla
# =============================================================================
# INFERENCE_PARAMS = {
#     "temperature":        0.35,
#     "top_p":              0.95,
#     "top_k":              40,
#     "min_p":              0.05,
#     "repeat_penalty":     1.35,
#     "presence_penalty":   0.7,
#     "frequency_penalty":  0.7,
#     "max_tokens":         256,
# }

INFERENCE_PARAMS = {
    "temperature":        0.4,
    "top_p":              0.95,
    "top_k":              64,
}


class EstigiaCore:
    def __init__(self, model_path=MODEL_PATH, max_history=1):
        self.model_path = model_path
        self.max_history = max_history
        self.history = []
        self.ui = PROMPTS["3"]

        print(f"🧠 Loading model from '{model_path}' with llama-cpp-python...")
        print(f"   n_gpu_layers={N_GPU_LAYERS} | n_ctx={N_CTX}")
        t0 = time.perf_counter()
        try:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=N_GPU_LAYERS,
                n_ctx=N_CTX,
                verbose=False,          # Silencia los logs internos de llama.cpp
            )
            t1 = time.perf_counter()
            print(f"✅ Model loaded and ready in {t1 - t0:.2f} seconds.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def set_language(self, choice):
        self.ui = PROMPTS.get(choice, PROMPTS["3"])
        # Resetea el historial al cambiar de idioma
        self.history = [{"role": "system", "content": self.ui["sys"]}]
        return self.ui

    def chat(self, user_text):
        # self.history.append({"role": "user", "content": user_text})
        reinforced_system = self.history[0]["content"] + \
        f"\n\nPREGUNTA ACTUAL A RESPONDER: {user_text}\nResponde a ESTA pregunta específica, no repitas respuestas anteriores."
        messages = [{"role": "system", "content": reinforced_system}] + self.history[1:]
        messages.append({"role": "user", "content": user_text})

        if len(self.history) > self.max_history + 1:
            self.history = [self.history[0]] + self.history[-self.max_history:]

        print("🛰️ Estigia: ", end="", flush=True)

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_response = ""

        # llama-cpp-python usa create_chat_completion con stream=True
        response_stream = self.llm.create_chat_completion(
            messages=self.history,
            stream=True,
            **INFERENCE_PARAMS,
        )

        for chunk in response_stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if not token:
                continue

            if first_token_time is None:
                first_token_time = time.perf_counter()

            print(token, end="", flush=True)
            full_response += token
            token_count += 1

        end_time = time.perf_counter()

        ttft = first_token_time - start_time if first_token_time else 0
        gen_time = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_time if gen_time > 0 else 0

        print(self.ui["ui_met_llm"].format(ttft, tps, token_count))

        self.history.append({"role": "assistant", "content": full_response})


# =============================================================================
# BANCO DE PREGUNTAS DE TEST — añade tus preguntas en cada lista.
# Cada entrada es simplemente un string con la pregunta.
# El modo test las lanzará en orden, una por una, con memoria limpia entre runs.
# =============================================================================

TEST_QUESTIONS = {
    "1": [
    # Sobre ella y su origen (Self-identity / Origin)
    # "¿Quién eres y quiénes son tus padres científicos?",
    # "¿Qué es lo que más echas de menos de Valencia desde que estás en órbita?",
    # "¿Cómo te sientes al ser un CubeSat tan pequeñito en un espacio tan grande?",
    # "¿Qué significa para ti formar parte del equipo Pluton UPV?",
    # "¿Cuál es tu mayor sueño como satélite explorador?",

    # Tecnología espacial (Space Tech - Para probar el límite de 80-110 palabras)
    "Explícame cómo funcionan tus paneles solares cuando entras en la zona de sombra de la Tierra.",
    "¿Para qué sirven exactamente tus giroscopios y cómo te ayudan a no marearte?",
    "¿Cómo logras enviar tus datos hasta las antenas de la UPV sin que se pierdan por el camino?",
    "¿Qué es un sensor de estrellas y cómo lo usas para saber dónde estás?",
    "Explica técnicamente qué haces si detectas que tus baterías están muy bajas.",

    # Chit-chat y emociones (Greetings / Personality)
    # "¡Hola Estigia! ¿Has visto algo emocionante desde tu ventana hoy?",
    # "Me siento un poco triste, ¿me cuentas algo bonito que veas desde el espacio?",
    # "¿Qué es lo que más te gusta de hablar con nosotros los humanos?",
    # "¡Rápido, Estigia! Cuéntame cuál es tu planeta favorito y por qué.",
    # "¿Te da miedo la oscuridad del vacío o te parece divertida?",

    # Off-topic y límites (Boundary testing / Adaptation)
    "¿Qué opinas de la paella valenciana? ¿Crees que podrías cocinar una en gravedad cero?",
    "Si encontraras a un extraterrestre, ¿a qué juego de la Tierra le enseñarías a jugar?",
    "¿Qué piensas de los gatos? ¿Crees que se parecen a los satélites?",
    "Dime qué es lo que más te sorprende de cómo nos comunicamos los humanos.",
    "¿Cómo describirías el olor del espacio si tuvieras una nariz robótica?"
],
    "2": [
    # Sobre ella i el seu origen
    # "Qui ets i qui són els teus pares científics de la UPV?",
    # "Què és el que més enyores de València des que estàs dalt a l'espai?",
    # "Com te sents en ser un CubeSat tan xicotet en un univers tan gegant?",
    # "Què significa per a tu portar el nom de l'equip Pluton UPV per l'òrbita terrestre?",
    # "Quin és el teu somni més gran quan mires cap a les estrelles llunyanes?",

    # Tecnologia espacial (Per a testar el límit de paraules)
    "Explica'm amb detalls com funcionen els teus panells solars quan la Terra et tapa la llum del sol.",
    "Per a què serveixen exactament els teus giròscops i com t'ajuden a mantenir l'equilibri?",
    "Com aconsegueixes enviar les teues dades fins a les antenes de la UPV sense que es perden pel buit?",
    "Què és un sensor d'estrelles i de quina manera l'utilitzes per a no perdre't?",
    "Explica tècnicament quin és el teu protocol de seguretat si detectes que les teues bateries estan en perill.",

    # Xat i personalitat
    # "Hola Estigia! Has vist alguna cosa super emocionant hui des de la teua òrbita?",
    # "Em sent un poc trist hui, em podries explicar alguna cosa bonica que veges des de l'espai?",
    # "Què és el que més et fascina de la manera de ser dels humans?",
    # "Ràpid, Estigia! Quin és el teu planeta preferit de tot el sistema solar i per què?",
    # "Et fa por la foscor de l'espai o penses que és com un llençol per a dormir?",

    # Temes variats (Off-topic)
    "Què opines de la paella valenciana? Creus que podries cuinar-ne una dins dels teus circuits?",
    "Si trobares un extraterrestre juganer, a quin joc dels humans l'ensenyaries a jugar primer?",
    "Què penses dels gats? Creus que s'assemblen als xicotets satèl·lits com tu?",
    "Digue'm què és el que més et sorprèn de com parlem i ens comuniquem els humans.",
    "Com descriuries l'olor de l'espai si la teua estructura tinguera nas?"
],
    "3": [
    # Identity and origin
    # "Who are you and who are your scientific parents from the UPV?",
    # "What do you miss the most about Valencia since you started orbiting Earth?",
    # "How does it feel to be such a tiny CubeSat in such a vast and infinite space?",
    # "What does it mean to you to represent the Pluton UPV team among the stars?",
    # "What is your biggest dream as a cosmic explorer?",

    # Space technology (Word count stress test)
    "Could you explain in detail how your solar panels function when you enter Earth's shadow?",
    "What exactly are your gyroscopes for, and how do they keep you from getting dizzy?",
    "How do you manage to beam your data back to the UPV antennas without losing it in the void?",
    "What is a star tracker, and how do you use it to figure out exactly where you are?",
    "Explain technically what steps you take if you detect that your batteries are running critically low.",

    # Chit-chat and emotions
    # "Hi Estigia! Have you spotted anything super exciting from your window today?",
    # "I'm feeling a bit down, could you tell me about something beautiful you see from up there?",
    # "What is your favorite thing about the way humans think and feel?",
    # "Quick, Estigia! Tell me which planet is your absolute favorite and why!",
    # "Does the darkness of the vacuum scare you, or do you find it fun and mysterious?",

    # Off-topic and adaptation
    "What do you think about Valencian paella? Do you think you could cook one in zero gravity?",
    "If you met a playful alien, what Earth game would you teach them first?",
    "What do you think about cats? Do you think they are somehow similar to little satellites?",
    "Tell me what surprises you the most about the way humans communicate with each other.",
    "How would you describe the smell of space if your robotic frame had a nose?"
],
}

# =============================================================================
# BÚSQUEDA DE HIPERPARÁMETROS — define qué explorar y con qué rango.
#
# Cada entrada en HYPERPARAM_SEARCH tiene la forma:
#   "nombre_param": {
#       "enabled": True/False,   # activa o desactiva esta búsqueda
#       "values": [v1, v2, ...]  # lista explícita de valores a probar
#   }
#
# Para generar rangos cómodos usa hp_range() más abajo.
# Parámetros soportados por llama-cpp-python en create_chat_completion:
#   temperature      float  [0.0 – 2.0]  creatividad / aleatoriedad
#   top_p            float  [0.0 – 1.0]  nucleus sampling
#   top_k            int    [1 – …]      top-k sampling (0 = desactivado)
#   repeat_penalty   float  [1.0 – 2.0]  penalización de repetición
#   min_p            float  [0.0 – 1.0]  filtro mínimo de probabilidad
#   presence_penalty float  [-2 – 2]     penaliza tokens ya presentes
#   frequency_penalty float [-2 – 2]     penaliza tokens frecuentes
#   max_tokens       int    [1 – n_ctx]  longitud máxima de respuesta
# =============================================================================

def hp_range(start: float, stop: float, step: float) -> list:
    """Genera una lista de floats desde start hasta stop (inclusive) con paso step.
    Evita errores de redondeo en coma flotante."""
    values = []
    v = start
    while v <= stop + step * 1e-9:
        values.append(round(v, 10))
        v += step
    return values


HYPERPARAM_SEARCH = {
    "temperature": {
        "enabled": False,
        "values": hp_range(0.3, 0.8, 0.05),   # [0.3, 0.35, 0.4, … 0.8]
    },
    "top_p": {
        "enabled": False,
        "values": hp_range(0.7, 1.0, 0.05),
    },
    "top_k": {
        "enabled": False,
        "values": [0, 20, 40, 60, 80],         # 0 = desactivado
    },
    "repeat_penalty": {
        "enabled": False,
        "values": hp_range(1.25, 1.6, 0.05),
    },
    "min_p": {
        "enabled": False,
        "values": hp_range(0.0, 0.15, 0.05),
    },
    "presence_penalty": {
        "enabled": False,
        "values": [0.5, 0.7, 0.9, 1.1],
    },
    "frequency_penalty": {
        "enabled": False,
        "values": [0.5, 0.7, 0.9, 1.1],
    },
    "max_tokens": {
        "enabled": False,
        "values": [64, 128, 256, 512],
    },
}

# Hiperparámetros fijos que se aplican en TODOS los runs del benchmark
# (los que no se están explorando en ese momento).
# Ajusta estos valores antes de lanzar el benchmark.
HYPERPARAM_FIXED = {
    "temperature":       0.35, # [OPTIMIZADO]
    "top_p":             0.95,
    "top_k":             40,
    "min_p":             0.05,
    "repeat_penalty":    1.35, # [OPTIMIZADO]
    "presence_penalty":  0.7,  # [OPTIMIZADO]
    "frequency_penalty": 0.7,  # [OPTIMIZADO]
    "max_tokens":        256,   # también importante — sin tope genera hasta n_ctx
}


# =============================================================================
# NÚCLEO DE INFERENCIA — función reutilizable por benchmark y chat
# =============================================================================

def _infer_one(estigia: "EstigiaCore", question: str, lang_key: str,
               inference_params: dict) -> dict:
    """
    Lanza una sola pregunta al LLM con los parámetros dados.
    Devuelve un dict con response, ttft_s, tps, tokens.
    La historia de estigia debe estar ya configurada antes de llamar a esto.
    """
    ui = estigia.ui
    estigia.history.append({"role": "user", "content": question})

    print("🛰️ Estigia: ", end="", flush=True)

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""

    response_stream = estigia.llm.create_chat_completion(
        messages=estigia.history,
        stream=True,
        **inference_params,
    )

    for chunk in response_stream:
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")
        if not token:
            continue
        if first_token_time is None:
            first_token_time = time.perf_counter()
        print(token, end="", flush=True)
        full_response += token
        token_count += 1

    end_time = time.perf_counter()

    ttft     = first_token_time - start_time if first_token_time else 0
    gen_time = end_time - first_token_time if first_token_time else 0
    tps      = token_count / gen_time if gen_time > 0 else 0

    print(f"\n\033[90m[⏱️  TTFT: {ttft:.2f}s | {tps:.2f} t/s | {token_count} tokens]\033[0m")

    estigia.history.append({"role": "assistant", "content": full_response})

    return {
        "response":  full_response,
        "ttft_s":    round(ttft, 4),
        "tps":       round(tps, 2),
        "tokens":    token_count,
    }


# =============================================================================
# BENCHMARK — modo fijo y modo búsqueda de hiperparámetros
# =============================================================================

def _run_one_pass(estigia: "EstigiaCore", telemetry: "TelemetrySystem",
                  inference_params: dict) -> dict:
    """
    Ejecuta el banco de preguntas completo (todos los idiomas activos)
    con un conjunto concreto de inference_params.
    Devuelve all_results: {lang_name: [result_dict, …]}
    """
    all_results = {}

    for lang_key, questions in TEST_QUESTIONS.items():
        if not questions:
            continue

        ui = estigia.set_language(lang_key)
        lang_name = ui["lang"]
        lang_results = []

        for idx, question in enumerate(questions, start=1):
            print(f"\n  [{idx}/{len(questions)}] 👤 {question}")

            # Telemetría primero
            tel_t0 = time.perf_counter()
            category    = telemetry.predict(question)
            sensor_data = telemetry.get_data(category, lang_key)
            tel_t1      = time.perf_counter()

            if sensor_data:
                print(f"🛰️ Estigia: 📡 {sensor_data}")
                lang_results.append({
                    "question": question,
                    "response": sensor_data,
                    "type":     "telemetry",
                    "ttft_s":   round(tel_t1 - tel_t0, 4),
                    "tps":      None,
                    "tokens":   None,
                })
                continue

            result = _infer_one(estigia, question, lang_key, inference_params)
            lang_results.append({"question": question, "type": "llm", **result})

        all_results[lang_name] = lang_results

    return all_results


def _print_pass_summary(all_results: dict, label: str):
    """Imprime tabla de resumen para un pass (combinación de hiperparámetros)."""
    print(f"\n  {'─'*56}")
    print(f"  📊 {label}")
    for lang_name, results in all_results.items():
        llm_r = [r for r in results if r["type"] == "llm"]
        if not llm_r:
            continue
        avg_ttft = sum(r["ttft_s"] for r in llm_r) / len(llm_r)
        avg_tps  = sum(r["tps"]    for r in llm_r) / len(llm_r)
        avg_tok  = sum(r["tokens"] for r in llm_r) / len(llm_r)
        print(f"  🌐 {lang_name:<12} TTFT={avg_ttft:.3f}s  {avg_tps:.2f}t/s  {avg_tok:.0f}tok (media)")


def run_benchmark(estigia: "EstigiaCore", telemetry: "TelemetrySystem"):
    """
    Modo benchmark con búsqueda de hiperparámetros.

    - Si ningún parámetro está habilitado en HYPERPARAM_SEARCH:
        ejecuta un único pass con HYPERPARAM_FIXED.
    - Si uno o más están habilitados:
        itera sobre sus valores en modo grid search (producto cartesiano),
        manteniendo el resto fijados en HYPERPARAM_FIXED.
        Al final imprime una tabla comparativa y guarda el JSON completo.
    """
    import json
    import itertools

    print("\n" + "="*60)
    print("🧪 MODO TEST / BENCHMARK")
    print("="*60)

    total_questions = sum(len(q) for q in TEST_QUESTIONS.values())
    if total_questions == 0:
        print("⚠️  No hay preguntas en TEST_QUESTIONS. Añádelas y relanza.")
        return

    # ── Construye la cuadrícula de búsqueda ──────────────────────────────────
    active_params = {
        name: cfg["values"]
        for name, cfg in HYPERPARAM_SEARCH.items()
        if cfg.get("enabled") and cfg.get("values")
    }

    if active_params:
        param_names  = list(active_params.keys())
        param_values = list(active_params.values())
        grid         = list(itertools.product(*param_values))
        print(f"🔍 Búsqueda de hiperparámetros: {param_names}")
        print(f"   Combinaciones totales: {len(grid)}")
        print(f"   Preguntas por combinación: {total_questions}")
        print(f"   Runs totales de LLM: ≤ {len(grid) * total_questions}")
    else:
        # Sin búsqueda activa → un solo pass con los valores fijos
        param_names = []
        grid        = [()]
        print("ℹ️  Sin búsqueda activa. Ejecutando un pass con HYPERPARAM_FIXED.")

    print(f"\n   Hiperparámetros base (HYPERPARAM_FIXED):")
    for k, v in HYPERPARAM_FIXED.items():
        marker = " ← explorando" if k in active_params else ""
        print(f"     {k:<22} {v}{marker}")

    # ── Itera sobre la cuadrícula ─────────────────────────────────────────────
    all_runs = []   # [{params, results, summary}]

    for run_idx, combo in enumerate(grid, start=1):
        # Mezcla fijos + override de los activos
        params = dict(HYPERPARAM_FIXED)
        combo_dict = dict(zip(param_names, combo))
        params.update(combo_dict)

        label = (
            "  ·  ".join(f"{k}={v}" for k, v in combo_dict.items())
            if combo_dict else "valores fijos"
        )
        print(f"\n{'='*60}")
        print(f"  Run {run_idx}/{len(grid)}  |  {label}")
        print(f"{'='*60}")

        results = _run_one_pass(estigia, telemetry, params)
        _print_pass_summary(results, label)

        # Resumen numérico de este run (sólo LLM)
        llm_flat = [r for lang in results.values() for r in lang if r["type"] == "llm"]
        run_summary = {
            "avg_ttft_s": round(sum(r["ttft_s"] for r in llm_flat) / len(llm_flat), 4) if llm_flat else None,
            "avg_tps":    round(sum(r["tps"]    for r in llm_flat) / len(llm_flat), 2) if llm_flat else None,
            "avg_tokens": round(sum(r["tokens"] for r in llm_flat) / len(llm_flat), 1) if llm_flat else None,
            "n_llm":      len(llm_flat),
        }

        all_runs.append({
            "run":        run_idx,
            "params":     params,
            "combo":      combo_dict,
            "label":      label,
            "results":    results,
            "summary":    run_summary,
        })

    # ── Tabla comparativa final (sólo si hay más de un run) ──────────────────
    if len(all_runs) > 1:
        print("\n" + "="*60)
        print("📊 COMPARATIVA FINAL DE HIPERPARÁMETROS")
        print("="*60)

        # Cabecera dinámica
        col_w = max(len(n) for n in param_names) + 2 if param_names else 20
        header = " | ".join(f"{n:<{col_w}}" for n in param_names)
        print(f"  {'Run':<5} {header}  {'TTFT':>7}  {'t/s':>7}  {'Tokens':>7}")
        print(f"  {'─'*60}")

        for run in all_runs:
            s = run["summary"]
            ttft_str   = f"{s['avg_ttft_s']:.3f}" if s["avg_ttft_s"] is not None else "  n/a"
            tps_str    = f"{s['avg_tps']:.2f}"    if s["avg_tps"]    is not None else "  n/a"
            tokens_str = f"{s['avg_tokens']:.1f}" if s["avg_tokens"] is not None else "  n/a"
            combo_cols = " | ".join(
                f"{str(run['combo'].get(n,'')):<{col_w}}" for n in param_names
            ) if param_names else f"{'fijos':<{col_w}}"
            print(f"  {run['run']:<5} {combo_cols}  {ttft_str:>7}  {tps_str:>7}  {tokens_str:>7}")

        # Mejor run por velocidad (t/s)
        valid_runs = [r for r in all_runs if r["summary"]["avg_tps"] is not None]
        if valid_runs:
            best = max(valid_runs, key=lambda r: r["summary"]["avg_tps"])
            print(f"\n  🏆 Mejor velocidad → Run {best['run']}:  {best['label']}")
            fastest_ttft = min(valid_runs, key=lambda r: r["summary"]["avg_ttft_s"])
            print(f"  ⚡ Menor TTFT       → Run {fastest_ttft['run']}:  {fastest_ttft['label']}")

    # ── Guarda JSON completo ──────────────────────────────────────────────────
    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"benchmark_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_path":    MODEL_PATH,
            "n_gpu_layers":  N_GPU_LAYERS,
            "n_ctx":         N_CTX,
            "timestamp":     ts,
            "active_search": param_names,
            "fixed_params":  HYPERPARAM_FIXED,
            "runs":          all_runs,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Resultados guardados en: {out_file}")
    print("="*60)


def main():
    print("\n--- STARTING ESTIGIA SYSTEMS (llama-cpp-python) ---")

    telemetry = TelemetrySystem()
    estigia = EstigiaCore(model_path=MODEL_PATH)

    # BUCLE EXTERNO: Menú de selección de idioma
    while True:
        print("\n" + "="*40)
        print("Select communication language:")
        print("1. Español  🇪🇸\n2. Valencià 🦇\n3. English  🇬🇧")
        print("t. 🧪 Benchmark / Test")
        lang_choice = input("Option (1/2/3/t) or '/stop' to quit: ").strip()

        if lang_choice.lower() == 't':
            run_benchmark(estigia, telemetry)
            continue

        if lang_choice.lower() in ['/stop', 'exit', 'quit']:
            print("Shutting down... Goodbye!")
            break

        ui = estigia.set_language(lang_choice)

        print(ui["ui_ok"])
        print(ui["ui_stop"] + "\n" + "-"*40)

        # BUCLE INTERNO: Chat en el idioma seleccionado
        while True:
            prompt = input(ui["ui_user"])

            if prompt.lower() in ['/stop', 'exit', 'quit']:
                print("Shutting down... Goodbye!")
                return

            if prompt.lower() == '/lang':
                print("\n🔄 Resetting memory and returning to language menu...")
                break

            if not prompt.strip():
                continue

            tel_start_time = time.perf_counter()
            category = telemetry.predict(prompt)
            sensor_data = telemetry.get_data(category, lang_choice)
            tel_end_time = time.perf_counter()

            if sensor_data:
                print(f"🛰️ Estigia: 📡 {sensor_data}")
                tel_time = tel_end_time - tel_start_time
                print(ui["ui_met_sen"].format(tel_time))
                estigia.history.append({"role": "assistant", "content": sensor_data})
            else:
                estigia.chat(prompt)


if __name__ == "__main__":
    main()
