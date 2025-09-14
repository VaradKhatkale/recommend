from django.apps import AppConfig

class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'

    def ready(self):
        # This imports the engine, causing it to load when the server starts
        from . import engine