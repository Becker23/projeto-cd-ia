# 🧠 Classificador de Textos: Humano vs. Inteligência Artificial

Este projeto desenvolve e avalia um classificador capaz de identificar se um texto foi escrito por **um humano** ou **gerado por um modelo de Inteligência Artificial**.  
A solução combina técnicas de **Processamento de Linguagem Natural (PLN)** e **Aprendizado de Máquina**, incluindo vetorização via **TF-IDF** e classificação utilizando **LinearSVC**, além de uma interface web interativa para demonstração e experimentação.

---

## 🎯 Objetivos

- Construir um **modelo supervisionado** capaz de distinguir textos humanos de textos gerados por IA.
- Desenvolver um processo **reprodutível** de coleta, preparação e modelagem dos dados.
- Disponibilizar uma **interface web** para análise e interação com o classificador.
- Criar um **jogo educativo**, onde o usuário tenta adivinhar se um texto é humano ou IA.

---

## 📚 Fontes de Dados

O dataset foi composto a partir de múltiplas fontes para garantir diversidade textual:

| Fonte                          | Tipo de escrita            | Propósito                      |
| ------------------------------ | -------------------------- | ------------------------------ |
| **Wikipédia**                  | Texto enciclopédico formal | Escrita humana confiável       |
| **Redações Nota 1000 do ENEM** | Texto argumentativo        | Escrita humana estruturada     |
| **Reviews do Mercado Livre**   | Texto informal espontâneo  | Escrita natural cotidiana      |
| **Modelos de IA (Gemini)**     | Reescritas automáticas     | Referência de texto artificial |

Foram criados pares correspondentes para cada tema:

_\_original.txt → texto escrito por humano
_\_ia.txt → versão reescrita por IA

---

## 🧠 Metodologia

1. **Pré-processamento de texto**

   - Normalização de acentuação e caracteres
   - Remoção de ruídos
   - Tokenização

2. **Vetorização**

   - Representação utilizando **TF-IDF**

3. **Modelos Avaliados**

   - **LinearSVC** (baseline eficiente e leve)
   - **BERT (Transformers)** como modelo contextual mais robusto

4. **Avaliação**
   - Conjunto de treino e teste com divisão estratificada
   - Métricas: _accuracy, precision, recall, f1-score_ e matriz de confusão

---

## 📈 Resultados Obtidos

### Modelo TF-IDF + LinearSVC

| Métrica           | Valor     |
| ----------------- | --------- |
| **Acurácia**      | **0.899** |
| F1-score (Humano) | 0.892     |
| F1-score (IA)     | 0.905     |

### Modelo BERT (Fine-Tuning)

| Métrica           | Valor     |
| ----------------- | --------- |
| **Acurácia**      | **0.956** |
| F1-score (Humano) | 0.953     |
| F1-score (IA)     | 0.959     |

> O modelo **BERT apresentou desempenho superior**, indicando melhor capacidade de capturar nuances linguísticas entre escrita humana e IA.

---

## 🖥️ Interface Web

A aplicação web desenvolvida com **Flask** permite:

| Função                   | Descrição                                                         |
| ------------------------ | ----------------------------------------------------------------- |
| **Verificação de texto** | O usuário cola um texto e recebe diagnóstico + probabilidade      |
| **Jogo "Humano x IA"**   | Mostra um trecho aleatório e o usuário tenta adivinhar sua origem |

### Executando a aplicação

python app.py

Depois acesse:
http://127.0.0.1:5000/

## 🛠️ Tecnologias Utilizadas

| Categoria            | Ferramentas                       |
| -------------------- | --------------------------------- |
| Linguagem            | Python 3.x                        |
| Backend Web          | Flask                             |
| Machine Learning     | Scikit-learn                      |
| Modelo Avançado      | BERT (Transformers - HuggingFace) |
| Manipulação de Dados | Pandas / NumPy                    |
| Interface            | HTML, CSS, JavaScript             |

## 👤 Autores

Carlos Augusto Freire Maia de Oliveira
RA: 21.00781-0

Cesar Augusto Bresciani Junior
RA: 21.00478-0

Enzo Leonardo Sabatelli de Moura
RA: 21.01535-0
