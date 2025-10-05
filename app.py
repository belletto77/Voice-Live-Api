from flask import Flask
import os

# Create Flask application instance
app = Flask(__name__)

@app.route('/')
def home():
    return 'Voice Live API - Flask entrypoint for Azure Web App'

@app.route('/health')
def health():
    return {'status': 'healthy', 'message': 'Flask app is running'}

if __name__ == '__main__':
    # Get port from environment variable or default to 8000
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
