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

class TelemetrySystem:
    def __init__(self, model_path='Models/telemetry_classifier.joblib'):
        print("⚙️ Loading telemetry classifier (joblib)...")
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.classifier = joblib.load(model_path)
            t1 = time.perf_counter()
            print(f"✅ Telemetry loaded in {t1 - t0:.2f} seconds.")
        except Exception as e:
            print(f"⚠️ Warning: {model_path} not found. Using keyword fallback. Error: {e}")
            self.classifier = None

    def predict(self, prompt):
        if self.classifier:
            return self.classifier.predict([prompt])[0]
        else:
            return "qa"

    def get_data(self, category, lang_choice):
        alt = random.uniform(150, 500)
        temp = random.uniform(15.0, 40.0)
        
        if category == 'altitude telemetry': 
            if lang_choice == "1": return f"Altitud actual: {alt:.1f} Km"
            if lang_choice == "2": return f"Altitud actual: {alt:.1f} Km" 
            if lang_choice == "3": return f"Current altitude: {alt:.1f} Km"
            
        if category == 'temperature telemetry': 
            if lang_choice == "1": return f"Temperatura interna: {temp:.1f} ºC"
            if lang_choice == "2": return f"Temperatura interna: {temp:.1f} ºC"
            if lang_choice == "3": return f"Internal temperature: {temp:.1f} ºC"
            
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

        response = ollama.chat(
            model=self.model_name,
            messages=self.history,
            stream=False)
        
        print(response['message']['content'])
            
        # end_time = time.perf_counter()
        
        # TTFT (Time To First Token) -> Tiempo de evaluación del prompt
        ttft = response.get('prompt_eval_duration', 0) / 1e9
        
        # Tokens generados
        token_count = response.get('eval_count', 0)
        
        # Tiempo de generación puro
        gen_time = response.get('eval_duration', 0) / 1e9
        
        # Tokens por segundo (t/s)
        tps = token_count / gen_time if gen_time > 0 else 0

        # Imprimir métricas
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