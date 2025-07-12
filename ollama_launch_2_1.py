import ollama
import time
from colorama import Fore, Back, Style
import re
import os
import random
import joblib

from transformers import MarianMTModel, MarianTokenizer
from lingua import Language, LanguageDetectorBuilder


#######################
###Translation Layer###
#######################

# Detectores de idioma
detector = LanguageDetectorBuilder.from_languages(Language.CATALAN, Language.SPANISH).build()

# Modelos de traducción
models = {
    'es_to_en': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en'),
    'en_to_es': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es'),
    'cat_to_en': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ca-en'),
    'en_to_cat': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ca')
}

# Tokenizadores
tokenizers = {
    'es_tokenizer': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en'),
    'en_tokenizer': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es'),
    'cat_tokenizer': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ca-en'),
    'en_tokenizer_val': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ca')
}

def translate(texts, model, tokenizer, language="en"):
    # Prepara el texto para la traducción
    template = lambda text: f">>{language}<< {text}" if language != "en" else text
    src_texts = [template(text) for text in texts]

    # Tokeniza los textos
    start_time = time.time()  # Start time for translation
    encoded = tokenizer(src_texts, return_tensors="pt", padding=True)

    # Genera la traducción
    translated = model.generate(**encoded)

    # Decodifica la salida
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    end_time = time.time()  # End time for translation
    translation_time = end_time - start_time

    return translated_texts, translation_time

def translation_to_en(prompt):
    # Detecta el idioma del texto
    language = detector.detect_language_of(prompt)
    # Elige el modelo y tokenizador según el idioma detectado
    if language == Language.CATALAN:
        model = models['cat_to_en']
        tokenizer = tokenizers['cat_tokenizer']
    elif language == Language.SPANISH:
        model = models['es_to_en']
        tokenizer = tokenizers['es_tokenizer']
    else:
        raise ValueError("Idioma no soportado o no detectado correctamente.")
    
    # Traduce el texto y mide el tiempo
    translated_text, translation_time = translate([prompt], model, tokenizer)
    return translated_text[0], language, translation_time


def translate_back_to_original_language(text, original_language):
    # Detecta el idioma original y traduce la respuesta al idioma original
    if original_language == Language.CATALAN:
        model = models['en_to_cat']
        tokenizer = tokenizers['en_tokenizer_val']
    elif original_language == Language.SPANISH:
        model = models['en_to_es']
        tokenizer = tokenizers['en_tokenizer']
    else:
        # Si el idioma no es ni catalán ni español, lo dejamos en inglés
        return text
    
    # Traducir de vuelta al idioma original
    translated_text, translation_time = translate([text], model, tokenizer, language=original_language)
    return translated_text


###########################
### INTENT CLASSIFIER LAYER###
###########################

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


###########################
### MODEL LANGUAGE LAYER###
###########################

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
        print(n, ' ', model.model)
    
    # Ask the user to select a model
    modeln = int(input('Select a model: ')) - 1
    modelname = ollama.list()['models'][modeln].model
    print('Selected model:', modelname)
    iniciar = True
    chat = True

    # Inner loop to handle the chat functionality
    while(chat):
        total_time = 0  # List to store time taken for each word/token generation
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
            
            print('Original prompt: ', prompt)

            # Measure time to translate the prompt
            translation_start_time = time.time()
            translated_prompt, original_language, translation_time = translation_to_en(prompt)
            translation_end_time = time.time()
            translation_duration = translation_end_time - translation_start_time

            print('Translated prompt: ', translated_prompt)
            print(f"Translation took {translation_duration:.4f} seconds.")

            # Measure time for category detection
            category_start_time = time.time()  # Start time for category detection
            category = telem_clf.predict([translated_prompt])[0]
            category_end_time = time.time()  # End time for category detection
            category_detection_time = category_end_time - category_start_time
            print(f"Category detection took {category_detection_time:.4f} seconds.")

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
                conversation_history.append({'role': 'user', 'content': translated_prompt})

                # Measure time for model response generation
                model_start_time = time.time()  # Start time for model response generation
                response = ollama.chat(
                    model=modelname,
                    messages=conversation_history,
                    stream=False,
                )
                model_end_time = time.time()  # End time for model response generation
                model_generation_time = model_end_time - model_start_time
                print(f"Model generation took {model_generation_time:.4f} seconds.")

                # Measure time for translation of the response
                translation_back_start_time = time.time()  # Start time for translating the response back
                final_response = translate_back_to_original_language(response.message.content, original_language)[0]
                translation_back_end_time = time.time()  # End time for translating the response back
                translation_back_duration = translation_back_end_time - translation_back_start_time
                print(f"Translation back to original language took {translation_back_duration:.4f} seconds.")
                
                total_time= (translation_duration + category_detection_time + model_generation_time + translation_back_duration)

                # Append model response to conversation history
                conversation_history.append({'role': 'assistant', 'content': final_response})

                # Print the final translated response to the user
                print("Final Response: ", final_response)

        print()
        print('Total respose generation time: ', total_time)
