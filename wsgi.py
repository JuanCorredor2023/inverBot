# wsgi.py
import os
from app import create_app

# Crea la app usando tu factory
app = create_app()

if __name__ == "__main__":
    # Usa la variable de entorno PORT (que Render inyecta) o 5000 por defecto
    port = int(os.environ.get("PORT", 5000))
    # Arranca escuchando en todas las interfaces
    app.run(host="0.0.0.0", port=port, debug=False)