# Detector de Texto Humano vs. IA

Este projeto implementa um detector de texto para identificar se um determinado conteúdo foi escrito por um humano ou gerado por Inteligência Artificial (IA). Ele utiliza técnicas de Processamento de Linguagem Natural (PLN) e Machine Learning, com um modelo baseado em TF-IDF e LinearSVC, e oferece uma interface web interativa para testes e um jogo de adivinhação.

## 🚀 Funcionalidades

*   **Scrapper de Conteúdo (Opcional):** Um script para coletar textos (ex: da Wikipédia) e gerar versões de IA (ex: usando Gemini ou outro LLM).
*   **Limpeza e Pré-processamento:** Funções para limpar e preparar os textos para o treinamento do modelo.
*   **Treinamento do Modelo:** Um pipeline de Machine Learning usando `TfidfVectorizer` e `LinearSVC` para classificar textos como "humano" ou "ia".
*   **API de Predição:** Um endpoint Flask para receber um texto e retornar a predição e a probabilidade.
*   **Interface Web (Frontend):**
    *   **Página Principal:** Onde o usuário pode colar um texto para ser analisado.
    *   **Jogo:** Um jogo interativo onde o usuário tenta adivinhar se um trecho de texto foi escrito por humano ou IA, com pontuação.

## ⚙️ Tecnologias Utilizadas

*   **Python 3.x**
*   **Flask:** Framework web para o backend.
*   **Scikit-learn:** Para o modelo de Machine Learning (TF-IDF, LinearSVC).
*   **Pandas:** Para manipulação de dados.
*   **Numpy:** Para operações numéricas.
*   **Regex:** Para limpeza de texto.

## 🛠️ Configuração do Ambiente

1.  **Clone o repositório (se aplicável):**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <nome_do_repositorio>
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install Flask scikit-learn pandas numpy
    ```
    (Se você tiver um `requirements.txt`, use `pip install -r requirements.txt`)

## 📋 Como Rodar o Projeto

Siga os passos abaixo na ordem para configurar e executar o projeto.

### 1. Coleta e Geração de Dados (Scrapping)

**Objetivo:** Obter textos originais (ex: Wikipedia) e suas versões geradas por IA.
**Saída:** Arquivos `.txt` na pasta `saida_wiki`.

1.  **Crie a pasta de saída:**
    ```bash
    mkdir saida_wiki
    ```

2.  **Execute seu script de scrapping (se tiver um):**
    Assumimos que você tem um script de scrapping que coleta os textos. Se você não tem um, precisará criar manualmente os arquivos `__original.txt` e `__ia.txt` na pasta `saida_wiki`.

    **Exemplo de como os arquivos devem ser nomeados na pasta `saida_wiki`:**
    ```
    saida_wiki/
    ├── artigo_sobre_ciencia__original.txt
    ├── artigo_sobre_ciencia__ia.txt
    ├── historia_da_arte__original.txt
    ├── historia_da_arte__ia.txt
    └── ...
    ```
    *   `__original.txt`: Contém o texto escrito por humanos (ex: da Wikipédia).
    *   `__ia.txt`: Contém a versão do mesmo tópico gerada por uma IA (ex: Gemini, ChatGPT).

    Se você tiver um script Python para isso (ex: `scrapper.py`), execute-o:
    ```bash
    python scrapper.py
    ```
    *Certifique-se de que o script `scrapper.py` salva os arquivos no formato e local esperados.*

### 2. Treinamento do Modelo

**Objetivo:** Treinar o modelo de Machine Learning para classificar os textos.
**Saída:** `modelo_tfidf_linearsvc.pkl`, `vectorizer.pkl`, `metrics.json`, `dataset_textos_limpo.csv` na pasta `saida_wiki`.

1.  **Execute o script de treinamento do modelo:**
    O script `modelo.py` irá ler os textos da pasta `saida_wiki`, limpar, treinar o modelo e salvar os artefatos necessários.
    ```bash
    python modelo.py
    ```

2.  **Verifique a saída:**
    Após a execução, você deve ver mensagens no console sobre o desempenho do modelo e os arquivos gerados na pasta `saida_wiki`.

### 3. Execução do Aplicativo Flask

**Objetivo:** Iniciar o servidor web para a interface de detecção e o jogo.

1.  **Execute o aplicativo Flask:**
    ```bash
    python app.py
    ```

2.  **Acesse a aplicação no navegador:**
    Abra seu navegador e vá para:
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

    *   Na página inicial, você pode colar textos para verificar se são IA ou humanos.
    *   Clique no botão "Jogar o Game" para ir para a página do jogo e testar suas habilidades.
    *   No jogo, use o botão "Voltar para o Início" para retornar à página principal.

## 📂 Estrutura do Projeto
├── app.py # Aplicação Flask principal com rotas e lógica.
├── modelo.py # Script para pré-processamento, treinamento e avaliação do modelo.
├── saida_wiki/ # Pasta onde os textos brutos e os artefatos do modelo são armazenados.
│ ├── artigo_exemplo__original.txt
│ ├── artigo_exemplo__ia.txt
│ ├── ...
│ ├── dataset_textos_limpo.csv # Dataset final limpo (gerado por modelo.py)
│ ├── modelo_tfidf_linearsvc.pkl # Modelo treinado (gerado por modelo.py)
│ ├── vectorizer.pkl # TF-IDF Vectorizer (gerado por modelo.py)
│ └── metrics.json # Métricas de avaliação do modelo (gerado por modelo.py)
├── static/
│ └── style.css # Estilos CSS da aplicação.
└── templates/
├── index.html # Página inicial da aplicação.
└── game.html # Página do jogo.

## 📈 Avaliação do Modelo

As métricas de avaliação do modelo são geradas pelo `modelo.py` e salvas em `saida_wiki/metrics.json`. Elas incluem:

*   **Acurácia:** Medida geral de desempenho.
*   **Matriz de Confusão:** Detalha os Verdadeiros Positivos, Falsos Positivos, Verdadeiros Negativos e Falsos Negativos para cada classe.
*   **`n_train` e `n_test`:** Número de amostras nos conjuntos de treino e teste.

A performance do modelo pode ser vista no console ao executar `modelo.py` ou inspecionando o arquivo `metrics.json`.

---
Feito por Carlos Augusto Freire Maia de Oliveira - 21.00781-0
