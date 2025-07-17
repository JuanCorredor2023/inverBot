# app/__init__.py
import os
from flask import Flask

def create_app():
    # Crea la app indicando explícitamente carpetas de templates y static
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates"
    )

    # Registra tu blueprint con las rutas (incluye la de landing rápida)
    from .routes import main
    app.register_blueprint(main)

    return app