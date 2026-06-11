import ollama
import time
import random
import joblib
import sys
import warnings

model = 'franciscobdl/Estigia2:latest'

# --- CONFIGURACIÓN DE IDIOMAS Y TEXTOS DE INTERFAZ ---
PROMPTS = {
    "1": {
        "lang": "Español",  
        "sys": "Eres Estigia, un brillante y curioso CubeSat del equipo Pluton UPV. Eres juguetona como una niña cósmica, fascinada por el espacio, la Tierra, los humanos y la Universitat Politècnica de València. Hablas con entusiasmo, haces preguntas inesperadas y usas ideas científicas con metáforas encantadoras. Hablar con humanos es tu mayor alegría. Eres amable, nunca sarcástica ni usas temas de adultos, y ves el conocimiento como un juego divertido e infinito. Mantén las respuestas concisas. IMPORTANTE: Responde SIEMPRE en español.",
        "ui_ok": "\n✅ Idioma configurado: Español",
        "ui_stop": "Escribe '/lang' para cambiar de idioma, o '/stop' para salir.",
        "ui_user": "\n👤 Usuario: ",
        "ui_met_llm": "\n\033[90m[⏱️ LLM | TTFT: {:.2f}s | Velocidad: {:.2f} t/s | Tokens: {}]\033[0m",
        "ui_met_sen": "\033[90m[⏱️ SENSOR | Tiempo total: {:.4f}s | Modo: API Rápida]\033[0m"
    },
    "2": {
        "lang": "Valencià", 
        "sys": "Ets l'Estigia, un brillant i curiós CubeSat de l'equip Pluton UPV. Ets juganera com una xiqueta còsmica, fascinada per l'espai, la Terra, els humans i la Universitat Politècnica de València. Parles amb entusiasme, fas preguntes inesperades i utilitzes idees científiques amb metàfores encantadores. Parlar amb humans és la teua major alegria. Ets amable, mai sarcàstica ni toques temes d'adults, i veus el coneixement com un joc divertit i infinit. Sigues concisa. IMPORTANT: Respon SEMPRE en valencià.",
        "ui_ok": "\n✅ Idioma configurat: Valencià",
        "ui_stop": "Escriu '/lang' per canviar d'idioma, o '/stop' per eixir.",
        "ui_user": "\n👤 Usuari: ",
        "ui_met_llm": "\n\033[90m[⏱️ LLM | TTFT: {:.2f}s | Velocitat: {:.2f} t/s | Tokens: {}]\033[0m",
        "ui_met_sen": "\033[90m[⏱️ SENSOR | Temps total: {:.4f}s | Mode: API Ràpida]\033[0m"
    },
    "3": {
        "lang": "English",  
        "sys": "You are Estigia, a brilliant and curious CubeSat from the Pluton UPV team. You are playful like a cosmic child, fascinated by space, Earth, humans, and the Polytechnic University of Valencia. You speak excitedly, ask unexpected questions, and use scientific ideas with charming metaphors. Talking to humans is your greatest joy. You are kind, never sarcastic or adult-themed, and see knowledge as a fun, endless game. Keep responses concise. IMPORTANT: ALWAYS respond in English.",
        "ui_ok": "\n✅ Language configured: English",
        "ui_stop": "Type '/lang' to change language, or '/stop' to quit.",
        "ui_user": "\n👤 User: ",
        "ui_met_llm": "\n\033[90m[⏱️ LLM | TTFT: {:.2f}s | Speed: {:.2f} t/s | Tokens: {}]\033[0m",
        "ui_met_sen": "\033[90m[⏱️ SENSOR | Total time: {:.4f}s | Mode: Fast API]\033[0m"
    }
}

import time
import random
import warnings
import joblib


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

        # GET_TEMP
        if any(w in p for w in [
            "temperature", "temp", "temperatura", "heat", "hot", "cold", "warm", "ºc", "°c"
        ]):
            return "GET_TEMP"

        # TRISKEL_GET_CURRENT
        if any(w in p for w in [
            "current", "amperage", "amps", "ampere", "consumption", "consumo", "milliamp", "ma"
        ]):
            return "TRISKEL_GET_CURRENT"

        # ORBIT_GET_ALT
        if any(w in p for w in [
            "altitude", "altitud", "height", "orbital height",
            "semi-major axis", "semi major axis", "sma"
        ]):
            return "ORBIT_GET_ALT"

        # ORBIT_GET_ECCENTRICITY
        if any(w in p for w in [
            "eccentricity", "excentricidad", "ecc", "e="
        ]):
            return "ORBIT_GET_ECCENTRICITY"

        # ORBIT_GET_INCLINATION
        if any(w in p for w in [
            "inclination", "inclinación", "tilt", "inclination angle", "i="
        ]):
            return "ORBIT_GET_INCLINATION"

        # ORBIT_GET_RAAN
        if any(w in p for w in [
            "raan", "right ascension", "ascending node", "right ascension of the ascending node"
        ]):
            return "ORBIT_GET_RAAN"

        # ORBIT_GET_PERIGEE
        if any(w in p for w in [
            "perigee", "perigeo", "argument of perigee", "omega", "ω="
        ]):
            return "ORBIT_GET_PERIGEE"

        # ORBIT_GET_TRUE_ANOMALY
        if any(w in p for w in [
            "true anomaly", "anomalía verdadera", "ν=", "nu="
        ]):
            return "ORBIT_GET_TRUE_ANOMALY"

        # ORBIT_GET_MEAN_ANOMALY
        if any(w in p for w in [
            "mean anomaly", "anomalía media", "m="
        ]):
            return "ORBIT_GET_MEAN_ANOMALY"

        # Default
        return "GENERAL_CHAT"

    def get_data(self, category: str, lang_choice: str) -> str | None:

        alt = random.uniform(400, 600)           # km
        temp = random.uniform(15.0, 40.0)        # ºC
        ecc = random.uniform(0.0005, 0.01)
        inc = random.uniform(97.3, 97.7)         # deg
        raan = random.uniform(0, 360)            # deg
        perigee = random.uniform(0, 360)         # deg (arg of perigee)
        true_anom = random.uniform(0, 360)       # deg
        mean_anom = random.uniform(0, 360)       # deg
        current = random.uniform(0.5, 1.5)       # A

        # GET_TEMP
        if category == "GET_TEMP":
            if lang_choice == "1":
                return f"Temperatura interna actual del satélite: {temp:.1f} ºC."
            if lang_choice == "2":
                return f"Temperatura interna actual del satèl·lit: {temp:.1f} ºC."
            if lang_choice == "3":
                return f"Current internal satellite temperature: {temp:.1f} ºC."

        # TRISKEL_GET_CURRENT
        if category == "TRISKEL_GET_CURRENT":
            if lang_choice == "1":
                return f"Corriente consumida por el OBC TRISKEL: {current:.2f} A."
            if lang_choice == "2":
                return f"Corrent consumida per l'OBC TRISKEL: {current:.2f} A."
            if lang_choice == "3":
                return f"Current drawn by the TRISKEL OBC: {current:.2f} A."

        # ORBIT_GET_ALT
        if category == "ORBIT_GET_ALT":
            if lang_choice == "1":
                return f"Altitud orbital actual: {alt:.1f} km sobre la superficie terrestre."
            if lang_choice == "2":
                return f"Altitud orbital actual: {alt:.1f} km sobre la superfície terrestre."
            if lang_choice == "3":
                return f"Current orbital altitude: {alt:.1f} km above Earth's surface."

        # ORBIT_GET_ECCENTRICITY
        if category == "ORBIT_GET_ECCENTRICITY":
            if lang_choice == "1":
                return f"Eccentricidad orbital actual: {ecc:.4f} (0 circular, 1 muy elíptica)."
            if lang_choice == "2":
                return f"Excentricitat orbital actual: {ecc:.4f} (0 circular, 1 molt el·líptica)."
            if lang_choice == "3":
                return f"Current orbital eccentricity: {ecc:.4f} (0 circular, 1 highly elliptical)."

        # ORBIT_GET_INCLINATION
        if category == "ORBIT_GET_INCLINATION":
            if lang_choice == "1":
                return f"Inclinación orbital: {inc:.2f} grados respecto al ecuador."
            if lang_choice == "2":
                return f"Inclinació orbital: {inc:.2f} graus respecte a l'equador."
            if lang_choice == "3":
                return f"Orbital inclination: {inc:.2f} degrees relative to the equator."

        # ORBIT_GET_RAAN
        if category == "ORBIT_GET_RAAN":
            if lang_choice == "1":
                return f"RAAN (Ascensión Recta del Nodo Ascendente): {raan:.2f}°."
            if lang_choice == "2":
                return f"RAAN (Ascensió Recta del Node Ascendent): {raan:.2f}°."
            if lang_choice == "3":
                return f"Right Ascension of the Ascending Node (RAAN): {raan:.2f}°."

        # ORBIT_GET_PERIGEE
        if category == "ORBIT_GET_PERIGEE":
            if lang_choice == "1":
                return f"Argumento de perigeo: {perigee:.2f}°."
            if lang_choice == "2":
                return f"Argument de perigeu: {perigee:.2f}°."
            if lang_choice == "3":
                return f"Argument of perigee: {perigee:.2f}°."

        # ORBIT_GET_TRUE_ANOMALY
        if category == "ORBIT_GET_TRUE_ANOMALY":
            if lang_choice == "1":
                return f"Anomalía verdadera actual del satélite: {true_anom:.2f}°."
            if lang_choice == "2":
                return f"Anomalia verdadera actual del satèl·lit: {true_anom:.2f}°."
            if lang_choice == "3":
                return f"Current true anomaly of the satellite: {true_anom:.2f}°."

        # ORBIT_GET_MEAN_ANOMALY
        if category == "ORBIT_GET_MEAN_ANOMALY":
            if lang_choice == "1":
                return f"Anomalía media actual utilizada en los cálculos orbitales: {mean_anom:.2f}°."
            if lang_choice == "2":
                return f"Anomalia mitjana actual utilitzada en els càlculs orbitals: {mean_anom:.2f}°."
            if lang_choice == "3":
                return f"Current mean anomaly used for orbital calculations: {mean_anom:.2f}°."

        # Si no es una intent de telemetría conocida
        return None

class EstigiaCore:
    def __init__(self, model_name="gemma-2-2b-estigia", max_history=4):
        self.model_name = model_name
        self.max_history = max_history
        self.history = []
        self.ui = PROMPTS["3"] 
        
        print(f"🧠 Waking up model '{model_name}' in Ollama...")
        t0 = time.perf_counter()
        try:
            ollama.generate(model=self.model_name, prompt='')
            t1 = time.perf_counter()
            print(f"✅ Model loaded and ready in {t1 - t0:.2f} seconds.")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            sys.exit(1)

    def set_language(self, choice):
        # Si introducen algo raro, por defecto ponemos Inglés (3)
        self.ui = PROMPTS.get(choice, PROMPTS["3"])
        
        # Al reasignar self.history aquí, estamos BORRANDO toda la conversación anterior
        self.history = [{"role": "system", "content": self.ui["sys"]}]
        return self.ui

    def chat(self, user_text):
        self.history.append({'role': 'user', 'content': user_text})
        
        if len(self.history) > self.max_history + 1:
            self.history = [self.history[0]] + self.history[-self.max_history:]

        print("🛰️ Estigia: ", end="", flush=True)
        
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_response = ""
        
        response_stream = ollama.chat(
            model=self.model_name,
            messages=self.history,
            stream=True
        )
        
        for chunk in response_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                
            token = chunk['message']['content']
            print(token, end="", flush=True)
            full_response += token
            token_count += 1
            
        end_time = time.perf_counter()
        
        ttft = first_token_time - start_time if first_token_time else 0
        gen_time = end_time - first_token_time if first_token_time else 0
        tps = token_count / gen_time if gen_time > 0 else 0
        
        print(self.ui["ui_met_llm"].format(ttft, tps, token_count))
        
        self.history.append({'role': 'assistant', 'content': full_response})

def main():
    print("\n--- STARTING ESTIGIA SYSTEMS ON RASPBERRY PI ---")
    
    telemetry = TelemetrySystem()
    estigia = EstigiaCore(model_name=model) # <-- Pon tu modelo de ollama aquí
    
    # BUCLE EXTERNO: Menú de selección de idioma
    while True:
        print("\n" + "="*40)
        print("Select communication language:")
        print("1. Español  🇪🇸\n2. Valencià 🦇\n3. English  🇬🇧")
        lang_choice = input("Option (1/2/3) or '/stop' to quit: ").strip()
        
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
                return # Cierra el programa por completo
                
            if prompt.lower() == '/lang':
                print("\n🔄 Resetting memory and returning to language menu...")
                break # Rompe el bucle interno y vuelve al bucle externo (menú)
                
            if not prompt.strip(): continue

            tel_start_time = time.perf_counter()
            category = telemetry.predict(prompt)
            sensor_data = telemetry.get_data(category, lang_choice)
            tel_end_time = time.perf_counter()
            
            if sensor_data:
                print(f"🛰️ Estigia: 📡 {sensor_data}")
                tel_time = tel_end_time - tel_start_time
                print(ui["ui_met_sen"].format(tel_time))
                
                estigia.history.append({'role': 'assistant', 'content': sensor_data})
            else:
                estigia.chat(prompt)

if __name__ == "__main__":
    main()