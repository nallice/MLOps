from app import app
from flask import render_template, request, jsonify
import pickle as pkl
import os
import numpy as np
from app.controllers import ml_treinado

modelo = pkl.load(open('modelo.pkl','rb'))

# Decorator: aplicar uma função em cima de outra
@app.route("/")
def verifica_api_online():
  return "API ONLINE v1.0", 200

# defaults: define um valor default pra variável, ou seja, se ela vier sem valor nenhum
# methods: os métodos http, você consegue definir como um parâmetro na rota
@app.route('/predict', methods=['POST'])
def predict():
  dados = request.get_json(force=True)

  predicao = modelo.predict(np.array([list(dados)]))
  resultado = predicao

  resposta = {'No-show': int(resultado)}
  return jsonify(resposta)