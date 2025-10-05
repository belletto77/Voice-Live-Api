from flask import Flask, send_from_directory
app = Flask(__name__)

@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)
