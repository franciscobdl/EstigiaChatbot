import ollama
import time
from colorama import Fore, Back, Style
import re
import os
import random
import joblib

telem_clf = joblib.load('Models/telemetry_classifier.joblib')

# Function to detect if the prompt is asking about temperature
def detect_temperature(prompt):
    keywords = [
      "temperature",
      "degrees",
      "weather",
      "thermometer",
      "hot",
      "cold",
      "ambient"
    ]
    prompt = prompt.lower()
    for keyword in keywords:
        if keyword in prompt:
            return True
    return False

# Function to detect if the prompt is asking about battery status
def detect_battery(prompt):
    battery_keywords = [
        "battery", 
        "level of the battery", "battery status", "battery level"
    ]
    prompt = prompt.lower()
    for keyword in battery_keywords:
        if keyword in prompt:
            return True
    return False
    
# Function to generate a random battery level response
def generate_battery():
    return f"My battery level is {random.randint(50, 100)}%"

# Function to generate a random temperature response
def generate_temperature():
    return f"My temperature is {random.uniform(15., 40.)}ºC"

def generate_altitude():
    return f"My altitude is {random.uniform(150, 500)}Km"

# Initialize conversation history
conversation_history = []

# Infinite loop to keep the program running
while(True):
    n = 0
    npalabras = 0

    # List available models
    print('Available models:')
    for model in ollama.list()['models']:
        n += 1
        print(n, ' ', model['name'])
    # Ask the user to select a model
    modeln = int(input('Select a model: ')) - 1
    modelname = ollama.list()['models'][modeln]['name']
    print('Selected model:', modelname)
    iniciar = True
    chat = True

    # Inner loop to handle the chat functionality
    while(chat):
        tiempos = []  # List to store time taken for each word/token generation
        stopGenerating = False
        if iniciar:
            # Initial prompt setup
            prompt = ''
            iniciar = False
            ollama.generate(model=modelname, prompt='')
        else:
            # Get user input
            prompt = input('User: ')
            if prompt == '/stop': 
                break
            
            # Detect if prompt matches specific telemetry and respond accordingly
            category = telem_clf.predict([prompt])[0]
            print(category)

            if category == 'altitude telemetry': 
                response = generate_altitude()
                print(response)
                conversation_history.append({'role': 'assistant', 'content': response})
                continue
            elif category == 'temperature telemetry': 
                response = generate_temperature()
                print(response)
                conversation_history.append({'role': 'assistant', 'content': response})
                continue
            elif category == 'qa':
                # Append user input to conversation history
                conversation_history.append({'role': 'user', 'content': prompt})

                # Validar que cada mensaje en conversation_history es un diccionario con las claves necesarias
                valid_messages = all(
                    isinstance(msg, dict) and 'role' in msg and 'content' in msg and isinstance(msg['content'], str)
                    for msg in conversation_history
                )

                if not valid_messages:
                    print("Error: conversation_history contiene un formato no válido.")
                    break

                # Call the model with the conversation history
                stream = ollama.chat(
                    model=modelname,
                    messages=conversation_history,
                    stream=True,
                )

                # Record the start time for response generation
                inicio = time.time()
                npalabras = 0
                response_content = ""
                
                for chunk in stream:
                    ahora = time.time()
                    print(chunk['message']['content'], end='', flush=True)
                    npalabras += 1
                    tiempos.append(ahora - inicio)
                    inicio = ahora
                    response_content += chunk['message']['content']

                    # Stop generating if more than 50 tokens and a sentence-ending character is found
                    if npalabras > 50 and bool(re.findall(r"[.!?\n:#]", chunk['message']['content'])):
                        stopGenerating = True
                    if stopGenerating: 
                        print(chunk['message']['content'])
                        break

                # Append model response to conversation history
                conversation_history.append({'role': 'assistant', 'content': response_content})

        print()
        print('Average word generation time: ', sum(tiempos[1:])/(len(tiempos)-1))
