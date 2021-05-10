"""index.py: the entry point of this dash app."""
from app.layouts import serve_layout
from application import app

app.layout = serve_layout

port = 80
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port, debug=True)
