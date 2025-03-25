import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import torch.nn as nn
from datetime import datetime

class GameRecommender:
    def __init__(self):
        self.genre_weights = defaultdict(float)
        self.platform_weights = defaultdict(float)
        self.game_vectors = {}
        self.game_info = {}
        self.scaler = StandardScaler()
        self.all_genres = set()
        self.all_platforms = set()
        
    def _get_year(self, release_date):
        if not release_date or release_date == "Fecha no disponible":
            return 2000
        try:
            return datetime.strptime(release_date, "%Y-%m-%d").year
        except:
            return 2000
    
    def _update_feature_sets(self, games_data):
        """Actualiza los conjuntos de géneros y plataformas"""
        for game in games_data:
            self.all_genres.update(game.get("genres", []))
            self.all_platforms.update(game.get("platforms", []))
    
    def _create_game_vector(self, game):
        """Crea un vector de características para un juego"""
        # Extraer características
        genres = set(game.get("genres", []))
        platforms = set(game.get("platforms", []))
        
        try:
            rating = float(game.get("rating", 0.0))
        except (ValueError, TypeError):
            rating = 0.0
            
        try:
            year = float(self._get_year(game.get("released")))
        except (ValueError, TypeError):
            year = 2000.0
        
        # Crear vector de características con dimensión fija
        features = []
        
        # Agregar géneros (one-hot encoding)
        features.extend([1.0 if g in genres else 0.0 for g in sorted(self.all_genres)])
        
        # Agregar plataformas (one-hot encoding)
        features.extend([1.0 if p in platforms else 0.0 for p in sorted(self.all_platforms)])
        
        # Agregar rating y año normalizados
        features.extend([
            rating / 5.0,  # Normalizar rating a [0,1]
            (year - 1990) / 40.0  # Normalizar año a [0,1] aproximadamente
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _rebuild_all_vectors(self):
        """Reconstruye todos los vectores de juegos para mantener dimensiones consistentes"""
        old_vectors = self.game_vectors
        self.game_vectors = {}
        for game_name, game_info in self.game_info.items():
            self.game_vectors[game_name] = self._create_game_vector(game_info)
    
    def update_model(self, games_data):
        """Actualiza el modelo con nuevos juegos"""
        if not games_data:
            return
            
        # Actualizar conjuntos de características
        old_genre_size = len(self.all_genres)
        old_platform_size = len(self.all_platforms)
        
        self._update_feature_sets(games_data)
        
        # Si hay nuevas características, reconstruir todos los vectores
        if len(self.all_genres) != old_genre_size or len(self.all_platforms) != old_platform_size:
            self._rebuild_all_vectors()
        
        # Crear o actualizar vectores de juegos nuevos
        for game in games_data:
            if game["name"] not in self.game_vectors:
                self.game_info[game["name"]] = game
                self.game_vectors[game["name"]] = self._create_game_vector(game)
    
    def get_recommendations(self, recent_games, num_recommendations=3):
        """Obtiene recomendaciones basadas en similitud de vectores"""
        if not recent_games or len(recent_games) < 2:  # Requerir al menos 2 juegos
            return []
        
        # Actualizar modelo con juegos recientes
        self.update_model(recent_games)
        
        # Obtener vector promedio de los juegos recientes
        recent_vectors = []
        for game in recent_games:
            if game["name"] in self.game_vectors:
                recent_vectors.append(self.game_vectors[game["name"]])
        
        if not recent_vectors:
            return []
        
        # Todos los vectores deberían tener la misma dimensión ahora
        user_profile = np.mean(recent_vectors, axis=0)
        
        # Calcular similitud con todos los juegos
        similarities = {}
        for game_name, vector in self.game_vectors.items():
            if game_name not in [g["name"] for g in recent_games]:
                similarity = cosine_similarity(
                    user_profile.reshape(1, -1),
                    vector.reshape(1, -1)
                )[0][0]
                similarities[game_name] = similarity
        
        # Ordenar por similitud y obtener los top N
        recommended_games = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_recommendations]
        
        # Convertir a lista de diccionarios con información completa
        recommendations = []
        for game_name, similarity in recommended_games:
            game_info = self.game_info[game_name].copy()
            game_info["similarity"] = f"{similarity:.0%}"
            recommendations.append(game_info)
        
        return recommendations

    def filter_games_by_category(self, category, min_rating=4.0):
        """Filtra juegos por categoría y rating mínimo"""
        filtered_games = []
        for game_name, game_info in self.game_info.items():
            if category in game_info.get("genres", []) and float(game_info.get("rating", 0.0)) >= min_rating:
                filtered_games.append({
                    "name": game_name,
                    "rating": game_info.get("rating", "Sin calificación"),
                    "genres": game_info.get("genres", []),
                    "platforms": game_info.get("platforms", []),
                    "released": game_info.get("released", "Fecha no disponible")
                })
        return sorted(filtered_games, key=lambda x: float(x["rating"]), reverse=True)

    def get_recommendations_by_category(self, category, num_recommendations=5):
        """Obtiene recomendaciones de juegos por categoría con buen rating"""
        # Primero filtramos los juegos por la categoría deseada
        filtered_games = self.filter_games_by_category(category)
        
        # Si no hay juegos en la categoría, retornamos una lista vacía
        if not filtered_games:
            return []
            
        # Convertimos los juegos filtrados a un formato adecuado para el modelo
        games_data = [{"name": game["name"], "genres": game["genres"], 
                      "rating": game["rating"], "released": game["released"]}
                     for game in filtered_games]
        
        # Actualizamos el modelo con los juegos filtrados
        self.update_model(games_data)
        
        # Tomamos los primeros juegos de la categoría como base para las recomendaciones
        base_games = filtered_games[:3]
        
        # Obtenemos las recomendaciones basadas en similitud
        recommendations = self.get_recommendations(base_games, num_recommendations)
        
        # Aseguramos que las recomendaciones tengan un rating mínimo
        recommendations = [game for game in recommendations 
                         if float(game.get("rating", 0.0)) >= 4.0]
        
        return recommendations
