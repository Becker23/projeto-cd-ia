import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import random

# Iniciando Flask
app = Flask(__name__)

# Caminho para os arquivos
# Lembre-se de ajustar este caminho se 'saida_wiki' não estiver no diretório C:\\Users\\carlo\\Desktop\\CD e IA\\
# Uma boa prática é usar caminhos relativos ou variáveis de ambiente para o caminho.
# Ex: TEXTS_DIR = os.path.join(os.path.dirname(__file__), 'saida_wiki')
TEXTS_DIR = 'C:\\Users\\carlo\\Desktop\\CD e IA\\saida_wiki'

# Carregar o modelo treinado
model_path = "modelo_tfidf_linearsvc.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Carregar o vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Função para prever se é IA ou Humano
def predict(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    prob = model.decision_function(text_vectorized)
    # Convertendo para probabilidade (sigmoide) se o modelo não for um classificador de probabilidade
    # Se o modelo LinearSVC já tem predict_proba, use-o diretamente.
    # Caso contrário, a sigmoide é uma heurística comum para transformar scores de decisão.
    # Note que para LinearSVC, decision_function dá a distância para o hiperplano.
    prob = 1 / (1 + np.exp(-prob))
    return prediction[0], prob[0]

# Função para ler arquivos de texto aleatoriamente e pegar as primeiras 30 palavras
def get_random_game_text():
    # Listar todos os arquivos na pasta
    all_files = os.listdir(TEXTS_DIR)
    
    # Filtrar arquivos de IA e Wikipedia
    ia_files = [f for f in all_files if "__ia.txt" in f]
    wiki_files = [f for f in all_files if "__original.txt" in f]
    
    # Escolher aleatoriamente um arquivo de cada tipo para garantir que temos ambos
    selected_ia_file = random.choice(ia_files)
    selected_wiki_file = random.choice(wiki_files)
    
    # Ler o conteúdo dos arquivos
    with open(os.path.join(TEXTS_DIR, selected_ia_file), 'r', encoding='utf-8') as f:
        ia_full_text = f.read()

    with open(os.path.join(TEXTS_DIR, selected_wiki_file), 'r', encoding='utf-8') as f:
        wiki_full_text = f.read()

    # Pegar as primeiras 30 palavras de cada texto
    ia_text_excerpt = ' '.join(ia_full_text.split()[:30]) + "..."
    wiki_text_excerpt = ' '.join(wiki_full_text.split()[:30]) + "..."
    
    # Decidir qual texto será mostrado ao usuário e qual será a resposta correta
    if random.choice([True, False]):
        displayed_text = wiki_text_excerpt
        correct_source = "humano" # Assumindo Wikipedia = humano
    else:
        displayed_text = ia_text_excerpt
        correct_source = "ia"
        
    return displayed_text, correct_source


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Texto não enviado!"}), 400
    
    text = data['text']
    label, probability = predict(text)
    return jsonify({'label': label, 'probability': round(probability * 100, 2)})

@app.route('/game')
def game():
    # Na primeira carga da página do jogo, buscaremos um texto inicial
    initial_text, initial_correct_answer = get_random_game_text()
    return render_template('game.html', initial_text=initial_text, initial_correct_answer=initial_correct_answer)

# NOVO ENDPOINT para buscar apenas os dados do próximo texto
@app.route('/get_next_text', methods=['GET'])
def get_next_text_data():
    displayed_text, correct_source = get_random_game_text()
    return jsonify({'text': displayed_text, 'correct_answer': correct_source})


@app.route('/check_answer', methods=['POST'])
def check_answer():
    user_answer = request.form['user_answer'] # Mudei para user_answer para clareza
    correct_answer = request.form['correct_answer']
    
    score_change = 0
    message = ""
    if user_answer == correct_answer:
        score_change = 1
        message = "Correto!"
    else:
        score_change = 0 # Mantém 0 se estiver errado
        message = f"Incorreto! A resposta correta era: {correct_answer.upper()}"
        
    return jsonify({'score_change': score_change, 'message': message})

if __name__ == '__main__':
    app.run(debug=True)