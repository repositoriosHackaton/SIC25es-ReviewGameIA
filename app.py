import requests
import streamlit as st
import os
import json
import csv
import re
import time
from dotenv import load_dotenv
from transformers import MarianMTModel, MarianTokenizer
import torch
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import spacy
from itertools import combinations
from responses import generate_game_response, generate_no_results_response, generate_end_conversation_response
import html
from game_recommender import GameRecommender

nlp = spacy.load("es_core_news_sm")

# Cargar las variables del archivo .env
load_dotenv()

# Obtener la clave de API de RAWG
RAWG_API_KEY = os.getenv("RAWG_API_KEY")
OCR_API_KEYS = os.getenv("OCR_API_KEYS", "").split(",")
OCR_API_KEYS = [key.strip() for key in OCR_API_KEYS if key.strip()]

# Cache para almacenar las últimas búsquedas y el recomendador
if 'last_searches' not in st.session_state:
    st.session_state.last_searches = []

if 'recommender' not in st.session_state:
    st.session_state.recommender = GameRecommender()

# Configurar el modelo de traducción MarianMT
model_name = "Helsinki-NLP/opus-mt-en-es"
model_path = "models/marianmt"

def load_model():
    """Cargar o descargar el modelo MarianMT."""
    try:
        # Intentar cargar el modelo localmente
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
        return tokenizer, model
    except:
        # Si no existe localmente, descargarlo y guardarlo
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Guardar el modelo y el tokenizador
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        return tokenizer, model

# Cargar el modelo al inicio
tokenizer, model = load_model()

# Crear la carpeta 'cache' si no existe
if not os.path.exists("cache"):
    os.makedirs("cache")

# Función para cargar o inicializar el caché
def load_cache():
    """Cargar el caché de traducciones desde un archivo."""
    cache_file = "cache/translations.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

# Función para guardar el caché
def save_cache(cache):
    """Guardar el caché de traducciones en un archivo."""
    cache_file = "cache/translations.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Error al guardar el caché: {e}")

# Cargar el caché al inicio
cache = load_cache()

def translate_text(text):
    """Traduce el texto del inglés al español usando MarianMT."""
    try:
        # Verificar si la traducción ya está en caché
        if text in cache:
            return cache[text]
            
        # Dividir el texto en párrafos usando los saltos de línea
        paragraphs = text.split('\n')
        translated_paragraphs = []
        
        # Traducir cada párrafo por separado
        for paragraph in paragraphs:
            if paragraph.strip():  # Solo traducir si el párrafo no está vacío
                inputs = tokenizer(paragraph, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translated_paragraph = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_paragraphs.append(translated_paragraph)
            else:
                translated_paragraphs.append('')  # Mantener los saltos de línea vacíos
        
        # Unir los párrafos traducidos con saltos de línea
        translated_text = '\n'.join(translated_paragraphs)
        
        # Guardar en caché
        cache[text] = translated_text
        save_cache(cache)
        
        return translated_text
    except Exception as e:
        st.error(f"Error al traducir el texto: {e}")
        return text

# Crear la carpeta 'data' si no existe
if not os.path.exists("data"):
    os.makedirs("data")

# Función para la animación de escritura
def typewriter_effect(text, delay=0.01):
    """Muestra el texto con un efecto de máquina de escribir."""
    placeholder = st.empty()  # Crear un espacio reservado para el texto
    current_text = ""
    for char in text:
        current_text += char
        placeholder.markdown(current_text)  # Actualizar el texto en el espacio reservado
        time.sleep(delay)  # Retraso entre caracteres

# Función para mejorar la imagen antes de enviarla al OCR
def enhance_image(image_bytes):
    try:
        with Image.open(image_bytes) as img:
            # Convertir a escala de grises
            img = img.convert("L")
            
            # Mejorar el contraste
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)  # Aumentar el contraste

            # Mejorar el brillo
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1.5)  # Aumentar el brillo

            # Reducir el ruido
            img = img.filter(ImageFilter.MedianFilter(size=3))

            # Guardar la imagen procesada en BytesIO
            enhanced_bytes = BytesIO()
            img.save(enhanced_bytes, format="JPEG")
            enhanced_bytes.seek(0)
            return enhanced_bytes
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

def extract_text_ocr_space(image_bytes):
    enhanced_bytes = enhance_image(image_bytes)
    if not enhanced_bytes:
        return "No se pudo procesar la imagen."

    headers = {"User-Agent": "Mozilla/5.0"}

    for api_key in OCR_API_KEYS:  # Intentar con cada API Key
        for language in ["spa", "eng"]:  # Primero español, luego inglés
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files={"image": enhanced_bytes},
                data={
                    "apikey": api_key,
                    "language": language,
                    "isOverlayRequired": False,
                    "filetype": "JPG",
                    "OCREngine": 2
                },
                headers=headers
            )

            if response.status_code != 200:
                print(f"Error en la API (HTTP {response.status_code}): {response.text}")
                continue  # Intentar con la siguiente clave API

            try:
                result = response.json()
                if isinstance(result, str):  # Si `result` es una cadena, convertirla a diccionario
                    result = eval(result)
            except (ValueError, SyntaxError):
                st.error(f"Respuesta no válida de OCR.space: {response.text}")
                continue

            if isinstance(result, dict) and "ParsedResults" in result:
                extracted_text = result["ParsedResults"][0].get("ParsedText", "").replace("\n", " ")
                return extracted_text if extracted_text else "No se detectó texto en la imagen."

        print(f"La clave API {api_key[:5]}... falló o alcanzó su límite.")  # Mostrar solo parte de la clave

    st.error("Todas las claves API fallaron o se agotaron.")
    return "No se detectó texto en la imagen."

# Función para extraer el nombre del juego y otros detalles en lenguaje natural
def extract_game_name(user_input):
    """Extrae el nombre del juego y otros detalles de la entrada del usuario."""
    # Traducir el texto al español si está en inglés
    translated_input = translate_text(user_input)
    
    # Patrones para detectar consultas comunes
    patterns = [
        r"(?:háblame de|dime información sobre|qué sabes de|quiero saber sobre|busca)\s+(.+)",
        r"(?:juegos de|juegos para)\s+(.+)",
        r"(?:juegos de)\s+(.+)\s+(?:lanzados en|del año)\s+(\d{4})",
        r"(?:juegos de)\s+(.+)\s+(?:en)\s+(.+)"  # Ejemplo: "juegos de acción en PlayStation"
    ]

    for pattern in patterns:
        match = re.search(pattern, translated_input, re.IGNORECASE)
        if match:
            return match.groups()  # Devuelve una tupla con los grupos capturados

    # Si no se encuentra ningún patrón, devolver la entrada completa
    return (translated_input.strip(),)

def interpret_query(user_query):
    """Analiza la consulta del usuario y extrae los filtros relevantes."""

    # Si user_query es una tupla, convertirla en una cadena
    if isinstance(user_query, tuple):
        user_query = " ".join(user_query)
    
    # Asegurarse de que user_query sea una cadena y convertirla a minúsculas
    user_query = str(user_query).lower()

    # Procesar el texto con spaCy
    doc = nlp(user_query)

    # Lista de palabras clave para género y plataforma
    genre_keywords = ["acción", "aventura", "estrategia", "rpg", "deportes", "carreras", "simulación", "misterio", "terror", "plataformas"]
    platform_keywords = [
        "playstation", "xbox", "pc", "nintendo", "switch", "steam", "mobile", "android", 
        "ios", "mac", "xbox 360", "playstation 3", "xbox 360 games store", "playstation network (ps3)", 
        "iphone", "ipad", "windows phone", "playstation vita", "wii u", "browser", "playstation network (vita)", 
        "xbox one", "playstation 4", "linux", "amazon fire tv", "new nintendo 3ds", "nintendo switch", 
        "xbox series x|s"
    ]

    # Palabras que no aportan valor para los filtros (palabras basura)
    stop_words = {"todos", "juegos", "dame", "quiero", "información", "podrías", "darme", "consultar", "de", "en", "sobre", "para", "con", "y", "la", "el", "los", "las", "un", "una", "que", "quisiera", "saber", "quiero", "hablame", "acerca", "del", "alrededor", "acerca"}

    # Inicializar el diccionario de filtros
    filters = {}

    # Filtrar las palabras relevantes (eliminamos las palabras vacías y de puntuación)
    filtered_words = [token.text for token in doc if token.text not in stop_words and not token.is_punct]

    # Buscar las palabras clave de género y plataforma
    for word in filtered_words:
        if word in genre_keywords:
            filters["genre"] = word
        elif word in platform_keywords:
            filters["platform"] = word
        elif re.match(r'\d{4}', word):  # Si es un año
            filters["release_year"] = word
        else:  # El resto se considera nombre del juego
            if "name" not in filters:
                filters["name"] = word
            else:
                filters["name"] += f" {word}"

    # Si no se ha asignado ningún nombre, asumimos que todo el texto es el nombre del juego
    if "name" not in filters:
        filters["name"] = " ".join(filtered_words)

    # Mostrar los filtros aplicados
    #print(f"Filtros aplicados: {filters}")
    
    return filters

def build_api_url(filters, api_key):
    base_url = f"https://api.rawg.io/api/games"
    query_params = []

    # Add filters to the query parameters
    if "name" in filters:
        query_params.append(f"search={filters['name']}")
    if "genre" in filters:
        query_params.append(f"genres={filters['genre']}")
    if "platform" in filters:
        query_params.append(f"platforms={filters['platform']}")
    if "release_year" in filters:
        query_params.append(f"dates={filters['release_year']}-01-01,{filters['release_year']}-12-31")

    # Combine base URL with query parameters
    if query_params:
        base_url += "?" + "&".join(query_params)

    return base_url

def save_game_info_csv(game):
    print(f"Tipo de 'game': {type(game)}")
    print(f"Contenido de 'game': {game}")
    # Verificar si 'game' es un diccionario o una lista de diccionarios
    if isinstance(game, dict):
        game_data = game  # Si es un diccionario, usarlo directamente
    elif isinstance(game, list) and len(game) > 0 and isinstance(game[0], dict):
        game_data = game[0]  # Si es una lista de diccionarios, tomar el primer diccionario
    else:
        raise TypeError(f"Se esperaba un diccionario o una lista de diccionarios, pero se recibió {type(game)}")

    # Definir las cabeceras que deben ser las mismas para todos los registros
    header = ['name', 'description', 'release_date', 'platforms']
    
    # Verificación segura de la lista 'platforms'
    platforms = game_data.get('platforms', [])
    if not isinstance(platforms, list):
        platforms = []  # Si 'platforms' no es una lista, se asigna una lista vacía

    # Crear el diccionario con los datos que se escribirán en el archivo CSV
    row = {
        'name': game_data.get('name', 'Nombre no disponible'),
        'description': game_data.get('description', 'Descripción no disponible'),
        'release_date': game_data.get('released', 'No disponible'),
        'platforms': ', '.join([platform.get('platform', {}).get('name', 'Desconocida') for platform in platforms if isinstance(platform, dict)])
    }

    # Ruta del archivo donde se guardarán los datos
    file_path = 'data/game_info.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Crear directorio si no existe

    # Comprobar si el archivo ya existe y tiene contenido
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    # Abrir el archivo en modo append ('a') para agregar nuevos registros
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        # Escribir el encabezado solo si el archivo no existe o está vacío
        if not file_exists:
            writer.writeheader()
        
        # Escribir la fila con los datos del juego
        writer.writerow(row)

    return file_path  # Retornar la ruta del archivo guardado (opcional)

def save_game_info_json(data):
    file_path = 'data/game_info.json'

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Si el archivo no existe o está vacío, crear uno nuevo con el contenido
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        # Leer el archivo existente y agregar los nuevos datos
        try:
            with open(file_path, 'r+', encoding='utf-8') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []  # Si el archivo está vacío o corrupto, iniciar una lista vacía

                # Agregar los nuevos datos a la lista existente
                existing_data.extend(data)

                # Volver a escribir el archivo con los datos actualizados
                file.seek(0)
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error al procesar el archivo JSON: {e}")

def get_game_info(user_input):
    """
    Obtiene la información del juego desde RAWG.io API
    """
    try:
        # URL base de la API de RAWG
        search_url = "https://api.rawg.io/api/games"
        
        # Parámetros de búsqueda
        params = {
            "key": RAWG_API_KEY,
            "search": user_input,
            "page_size": 5
        }
        
        # Realizar la búsqueda
        response = requests.get(search_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data["count"] > 0:
                game = data["results"][0]  # Tomamos el primer resultado
                
                # Obtener detalles completos del juego usando su ID
                game_id = game["id"]
                details_url = f"https://api.rawg.io/api/games/{game_id}"
                details_params = {
                    "key": RAWG_API_KEY
                }
                details_response = requests.get(details_url, params=details_params)
                
                if details_response.status_code == 200:
                    game_details = details_response.json()
                    
                    # Limpiar la descripción de etiquetas HTML
                    description = game_details.get("description", "No hay descripción disponible.")
                    description = re.sub(r'<br\s*/?>|<p>|</p>', '\n', description)  # Reemplazar <br/>, <p> con saltos de línea
                    description = re.sub(r'<[^>]+>', '', description)  # Eliminar otras etiquetas HTML
                    description = html.unescape(description)  # Convertir entidades HTML
                    description = re.sub(r'\n\s*\n', '\n\n', description)  # Eliminar líneas vacías múltiples
                    description = description.strip()  # Eliminar espacios en blanco al inicio y final
                    
                    # Traducir el texto usando MarianMT
                    translated_description = translate_text(description)
                    
                    # Preparar los datos del juego
                    game_info = {
                        "id": game_id,
                        "name": game_details.get("name", "Nombre no disponible"),
                        "description": translated_description,  # Usar la descripción traducida
                        "rating": game_details.get("rating", 0),
                        "rating_count": game_details.get("ratings_count", 0),
                        "released": game_details.get("released", "Fecha no disponible"),
                        "platforms": [p["platform"]["name"] for p in game_details.get("platforms", [])],
                        "genres": [g["name"] for g in game_details.get("genres", [])],
                        "developers": [d["name"] for d in game_details.get("developers", [])],
                        "publishers": [p["name"] for p in game_details.get("publishers", [])],
                        "background_image": game_details.get("background_image", ""),
                        "metacritic": game_details.get("metacritic", None),
                        "esrb_rating": game_details.get("esrb_rating", {}).get("name", None)
                    }
                    
                    # Guardar la información del juego
                    save_game_info_json([game_info])
                    save_game_info_csv(game_info)
                    
                    return game_info
                else:
                    st.error(f"Error al obtener los detalles del juego: {details_response.status_code}")
            else:
                st.error("No se encontraron juegos con ese nombre.")
        else:
            st.error(f"Error en la búsqueda: {response.status_code}")
    except Exception as e:
        st.error(f"Error al obtener la información del juego: {str(e)}")
    return None

def show_recommendations():
    """Muestra las recomendaciones en la barra lateral"""
    if st.session_state.last_searches:
        st.sidebar.markdown("### Últimas búsquedas")
        for game in st.session_state.last_searches:
            st.sidebar.write(f"- {game['name']}")
        
        st.sidebar.markdown("### Recomendaciones")
        recommendations = st.session_state.recommender.get_recommendations(st.session_state.last_searches)
        
        if recommendations:
            for game in recommendations:
                st.sidebar.write(f"- {game['name']} ({game['similarity']} similar)")
                st.sidebar.write(f"  Géneros: {', '.join(game['genres'])}")

def display_game_info(game_info):
    # Mostrar imagen del juego
    if game_info["background_image"]:
        st.image(game_info["background_image"], caption=game_info["name"])
    
    # Información principal
    st.header("Información del Juego")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"🎯 **Nombre:** {game_info['name']}")
        st.write(f"📅 **Fecha de lanzamiento:** {game_info['released']}")
        st.write(f"⭐ **Rating:** {game_info['rating']}/5")
        if 'metacritic' in game_info and game_info['metacritic'] is not None:
            st.write(f"🎯 **Metacritic:** {game_info['metacritic']}")
    
    with col2:
        st.write(f"🎮 **Plataformas:** {', '.join(game_info['platforms'])}")
        st.write(f"🎲 **Géneros:** {', '.join(game_info['genres'])}")
        if 'esrb_rating' in game_info and game_info['esrb_rating'] is not None:
            st.write(f"📝 **ESRB Rating:** {game_info['esrb_rating']}")
    
    # Descripción
    st.subheader("Descripción")
    st.markdown(game_info["description"])
    
    # Recomendaciones basadas en géneros similares
    if len(st.session_state.last_searches) >= 3:
        st.subheader("Recomendaciones basadas en tus búsquedas")
        genres_count = {}
        for search in st.session_state.last_searches:
            for genre in search["genres"]:
                genres_count[genre] = genres_count.get(genre, 0) + 1
        
        most_common_genres = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)[:2]
        if most_common_genres:
            st.write(f"Basado en tus búsquedas, te gustan los juegos de {', '.join([g[0] for g in most_common_genres])}.")

def main():
    st.title("🎮 Asistente de Juegos")
    st.write("¡Hola! Soy tu asistente para encontrar información sobre juegos. Puedes preguntarme sobre cualquier juego.")
    
    # Inicializar el estado de la sesión
    if 'last_searches' not in st.session_state:
        st.session_state.last_searches = []
    if 'recommender' not in st.session_state:
        st.session_state.recommender = GameRecommender()
    
    # Sidebar para mostrar recomendaciones y búsquedas recientes
    with st.sidebar:
        st.header("🕹️ Panel de Jugador")
        if st.session_state.last_searches:
            st.subheader("🎯 Últimas Búsquedas")
            for game in st.session_state.last_searches[-3:]:
                st.write(f"🎮 {game['name']}")
            
            # Actualizar modelo y mostrar recomendaciones
            if len(st.session_state.last_searches) >= 2:
                st.subheader("🎮 Juegos Recomendados")
                # Actualizar el modelo con todas las búsquedas
                st.session_state.recommender.update_model(st.session_state.last_searches)
                recommendations = st.session_state.recommender.get_recommendations(st.session_state.last_searches)
                
                if recommendations:
                    for game in recommendations:
                        st.write(f"🎲 {game['name']} ({game['similarity']} similar)")
                        st.write(f"   Géneros: {', '.join(game['genres'])}")
                else:
                    st.info("Busca más juegos para obtener recomendaciones")
    
    # Input del usuario
    user_input = st.text_input("Escribe el nombre del juego que buscas:")
    uploaded_file = st.file_uploader("O sube una imagen con el nombre del juego", type=["png", "jpg", "jpeg"])

    if user_input or uploaded_file:
        if user_input.lower() in ["adiós", "adios", "chao", "hasta luego", "bye"]:
            response = generate_end_conversation_response()
            st.write(response)
        elif uploaded_file:
            with st.spinner("Procesando imagen..."):
                image_bytes = BytesIO(uploaded_file.getbuffer())
                extracted_text = extract_text_ocr_space(image_bytes)
                if extracted_text.strip():
                    game_info = get_game_info(extracted_text.strip())
                    if game_info:
                        st.success(generate_game_response(game_info["name"]))
                        
                        # Mostrar información del juego
                        display_game_info(game_info)
                        
                        # Actualizar últimas búsquedas
                        if not any(game['name'] == game_info['name'] for game in st.session_state.last_searches):
                            st.session_state.last_searches.append(game_info)
                            if len(st.session_state.last_searches) > 3:
                                st.session_state.last_searches.pop(0)
                            
                            # Actualizar modelo inmediatamente
                            st.session_state.recommender.update_model([game_info])
                    else:
                        st.warning(generate_no_results_response())
                else:
                    st.warning("No se pudo detectar texto en la imagen.")
                    st.write(generate_end_conversation_response())
        else:
            game_info = get_game_info(user_input)
            
            if game_info:
                st.success(generate_game_response(game_info["name"]))
                
                # Mostrar información del juego
                display_game_info(game_info)
                
                # Actualizar últimas búsquedas
                if not any(game['name'] == game_info['name'] for game in st.session_state.last_searches):
                    st.session_state.last_searches.append(game_info)
                    if len(st.session_state.last_searches) > 3:
                        st.session_state.last_searches.pop(0)
                    
                    # Actualizar modelo inmediatamente
                    st.session_state.recommender.update_model([game_info])
            else:
                st.warning(generate_no_results_response())

if __name__ == "__main__":
    main()