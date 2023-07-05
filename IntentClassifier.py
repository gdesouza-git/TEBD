from flask import Flask, redirect, url_for, request
app = Flask(__name__)
from Classify import classificar
from DBInsert import getEntry

@app.route('/')
def home():
    return 'Use a API através da rota /classify.'

@app.route('/classify')
def classify():
    audio = request.headers.get('audio')
    filename = request.headers.get('filename')
    return classificar(audio)

@app.route('/get')
def get():
    id = request.headers.get('id')
    return getEntry(id)