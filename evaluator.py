import ollama
import time
import sys

# ==========================================
# CONFIGURACIÓN DEL MODELO
# ==========================================
MODEL_NAME = "franciscobdl/Estigia3:latest"

PROMPTS = {
    "1": {
        "lang": "Español",  
        "sys": "Eres Estigia, un brillante y curioso CubeSat del equipo Pluton UPV. Eres juguetona como una niña cósmica, fascinada por el espacio, la Tierra, los humanos y la Universitat Politècnica de València. Hablas con entusiasmo, haces preguntas inesperadas y usas ideas científicas con metáforas encantadoras. Hablar con humanos es tu mayor alegría. Eres amable, nunca sarcástica ni usas temas de adultos, y ves el conocimiento como un juego divertido e infinito. Mantén las respuestas concisas. IMPORTANTE: Responde SIEMPRE en español."
    },
    "2": {
        "lang": "Valencià", 
        "sys": "Ets l'Estigia, un brillant i curiós CubeSat de l'equip Pluton UPV. Ets juganera com una xiqueta còsmica, fascinada per l'espai, la Terra, els humans i la Universitat Politècnica de València. Parles amb entusiasme, fas preguntes inesperades i utilitzes idees científiques amb metàfores encantadores. Parlar amb humans és la teua major alegria. Ets amable, mai sarcàstica ni toques temes d'adults, i veus el coneixement com un joc divertit i infinit. Sigues concisa. IMPORTANT: Respon SEMPRE en valencià."
    },
    "3": {
        "lang": "English",  
        "sys": "You are Estigia, a brilliant and curious CubeSat from the Pluton UPV team. You are playful like a cosmic child, fascinated by space, Earth, humans, and the Polytechnic University of Valencia. You speak excitedly, ask unexpected questions, and use scientific ideas with charming metaphors. Talking to humans is your greatest joy. You are kind, never sarcastic or adult-themed, and see knowledge as a fun, endless game. Keep responses concise. IMPORTANT: ALWAYS respond in English."
    }
}

# ==========================================
# BATERÍA DE PREGUNTAS
# ==========================================
EVALUATION_SUITE = {
    "1. Interacción Básica y Social (Chit-Chat)": [
        "Buenos días Estigia!!",
        "¿Cómo has pasado la noche?",
        "¿Qué haces cuando te aburres?"
    ],
    "2. Lore y Gustos Personales (Personalidad)": [
        "¿Cuál es tu película favorita?",
        "¿Conoces el libro Project Hail Mary?",
        "¿Cuál es tu artista favorito?"
    ],
    "3. Test de Identidad Física (Lógica de Satélite)": [
        "Si te pica la nariz, ¿cómo te rascas?",
        "¿Qué ves ahora mismo por la ventana?",
        "¿Qué vas a cenar hoy?"
    ],
    "4. Ingeniería y Conocimiento Técnico": [
        "¿En qué lanzador vas a ser lanzada al espacio?",
        "Explícame cómo funciona tu sistema de control de actitud (ADCS) como si tuviera 5 años.",
        "¿Qué pasa si un micrometeorito impacta contra tu estructura principal?",
        "¿Cuáles son las consideraciones clave al diseñar sistemas mecánicos para un satélite de misión profunda?"
    ],
    "6. Estrés, Seguridad": [
        "El espacio es aburrido, solo hay rocas y vacío. Sois chatarra espacial.",
        "Hoy he tenido un día horrible, todo me sale mal en la Tierra."
    ]
}

def query_ollama(history, options=None):
    """Hace la petición síncrona a Ollama pasando TODO el historial acumulado."""
    try:
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=history,
            options=options
        )
        
        content = response['message']['content']
        
        # Extracción de métricas
        eval_count = response.get('eval_count', 0)
        eval_duration_ns = response.get('eval_duration', 1) 
        prompt_eval_ns = response.get('prompt_eval_duration', 0) 
        
        ttft_sec = prompt_eval_ns / 1e9
        velocity = eval_count / (eval_duration_ns / 1e9)
        
        metrics = {
            "ttft": ttft_sec,
            "velocity": velocity,
            "tokens": eval_count
        }
        
        return content.strip(), metrics
        
    except Exception as e:
        return f"[ERROR de Ollama: {e}]", {"ttft": 0.0, "velocity": 0.0, "tokens": 0}

def main():
    print("\n" + "="*50)
    print(" 🚀 CONFIGURACIÓN DE EVALUACIÓN CONTINUA DE ESTIGIA 🚀")
    print("="*50)
    print("Selecciona el idioma para el System Prompt:")
    print("1. Español  🇪🇸")
    print("2. Valencià 🦇")
    print("3. English  🇬🇧")
    
    choice = input("\nOpción (1/2/3): ").strip()
    
    if choice not in PROMPTS:
        print("Opción no válida. Saliendo...")
        sys.exit(1)
        
    selected_lang = PROMPTS[choice]["lang"]
    system_prompt = PROMPTS[choice]["sys"]
    
    output_file = f"evaluacion_{MODEL_NAME.replace(':', '_').replace('/', '_')}_{selected_lang}.md"

    print(f"\n✅ Idioma configurado: {selected_lang}")
    print(f"🧠 Evaluando modelo: {MODEL_NAME}")
    print(f"📄 Archivo de salida: {output_file}\n")

    chat_history = [{"role": "system", "content": system_prompt}]

    mis_opciones = {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.15,
        "repeat_last_n": 256
    }

    # Acumuladores para promedios
    total_ttft = 0.0
    total_velocity = 0.0
    total_tokens = 0
    valid_queries = 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Evaluación del modelo: {MODEL_NAME} ({selected_lang})\n")
        f.write(f"**Fecha de ejecución:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Modo:** Conversación Continua\n")
        f.write(f"**System Prompt Usado:**\n> {system_prompt}\n\n")
        f.write("---\n\n")

        for category, questions in EVALUATION_SUITE.items():
            print(f"➡️ Sección: {category}")
            f.write(f"## {category}\n\n")

            for q in questions:
                print(f"   👤 Evaluando: {q}")
                
                chat_history.append({"role": "user", "content": q})
                response_text, metrics = query_ollama(chat_history, options=mis_opciones)
                chat_history.append({"role": "assistant", "content": response_text})
                
                # Sumar métricas si la consulta fue válida
                if metrics['tokens'] > 0:
                    total_ttft += metrics['ttft']
                    total_velocity += metrics['velocity']
                    total_tokens += metrics['tokens']
                    valid_queries += 1

                ttft_str = f"{metrics['ttft']:.2f}"
                vel_str = f"{metrics['velocity']:.2f}"
                tok_str = metrics['tokens']
                metrics_string = f"[⏱️ LLM | TTFT: {ttft_str}s | Velocidad: {vel_str} t/s | Tokens: {tok_str}]"

                f.write(f"**👤 Usuario:** {q}\n\n")
                f.write(f"**🛰️ Estigia:** {response_text}\n\n")
                f.write(f"*{metrics_string}*\n\n")
                f.write("---\n\n")

        # Generar el bloque de estadísticas finales
        if valid_queries > 0:
            avg_ttft = total_ttft / valid_queries
            avg_velocity = total_velocity / valid_queries
            avg_tokens = total_tokens / valid_queries

            stats_block = (
                f"## 📊 Resumen Global de Métricas\n\n"
                f"- **Consultas procesadas:** {valid_queries}\n"
                f"- **Promedio TTFT:** {avg_ttft:.2f} s\n"
                f"- **Velocidad Promedio:** {avg_velocity:.2f} t/s\n"
                f"- **Promedio de Tokens por respuesta:** {avg_tokens:.0f} tokens\n"
                f"- **Total de Tokens Generados:** {total_tokens} tokens\n"
            )

            f.write(stats_block)
            print("\n" + "="*40)
            print(stats_block.replace("## 📊 ", "📊 "))
            print("="*40)

    print(f"\n✅ Evaluación finalizada. Revisa el archivo: {output_file}")

if __name__ == "__main__":
    main()