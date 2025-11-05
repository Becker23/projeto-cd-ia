# 🧠 Classificador de Textos: Humano vs. Inteligência Artificial

Este projeto desenvolve e avalia um classificador capaz de identificar se um texto foi escrito por **um humano** ou **gerado por um modelo de Inteligência Artificial**.  
A solução combina técnicas de **Processamento de Linguagem Natural (PLN)** e **Aprendizado de Máquina**, incluindo vetorização via **TF-IDF** e classificação utilizando **LinearSVC**, e o modelo **BERTimbau**, um modelo BERT pré-treinado com textos da língua portuguesa, além de uma interface web interativa para demonstração e experimentação.

**Para um relatório mais detalhado, leia o arquivo "Projeto_Semestral_CD.ipynb" disponível neste repositório ou [neste link](https://colab.research.google.com/drive/1AcidKOIanc64XdTVEW6PTRkf4c5L84Tw?usp=sharing#scrollTo=hcqBUpwTcX5A)**

**Os modelos utilizados neste projeto não estão no repositório por excederem o limite de tamanho do GitHub. Sendo assim, baixe o arquivo [models.zip](https://drive.google.com/file/d/1eONooP6zkEe-VbriwtVH_Ja0qP149ZFP/view?usp=sharing) do Google Drive. Após baixar, extraia a pasta "models" (a pasta inteira, não os arquivos) dentro de app/backend/.**

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

Para cada tipo de fonte, foram gerados datasets com textos tanto gerados por humanos quanto gerados por Inteligência Artificial.

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

5. **Cross-validation**
   - Avaliação de acurácia do modelo em diferentes conjuntos de treinamento e teste
   - Avaliação de possível _over-fitting_ do modelo

---

## 📈 Resultados Obtidos

### Modelo TF-IDF + LinearSVC

| Métrica           | Valor     |
| ----------------- | --------- |
| **Acurácia**      | **0.814** |
| F1-score (Humano) | 0.810     |
| F1-score (IA)     | 0.817     |

### Modelo BERT (Fine-Tuning)

| Métrica           | Valor     |
| ----------------- | --------- |
| **Acurácia**      | **0.956** |
| F1-score (Humano) | 0.953     |
| F1-score (IA)     | 0.958     |

> O modelo **BERT apresentou desempenho superior**, indicando melhor capacidade de capturar nuances linguísticas entre escrita humana e IA.

---

## Rodar aplicação

### Frontend

```
cd frontend
npm install
npm run dev
```

### Backend

```
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8080
```

## 🖥️ Interface Web

A aplicação web permite:

| Função                   | Descrição                                                         |
| ------------------------ | ----------------------------------------------------------------- |
| **Verificação de texto** | O usuário cola um texto e recebe diagnóstico + probabilidade      |
| **Jogo "Humano x IA"**   | Mostra um trecho aleatório e o usuário tenta adivinhar sua origem |

## 🛠️ Tecnologias Utilizadas

| Categoria            | Ferramentas                       |
| -------------------- | --------------------------------- |
| Linguagem            | Python 3.x                        |
| Backend Web          | Python com fast api               |
| Machine Learning     | Scikit-learn                      |
| Modelo Avançado      | BERT (Transformers - HuggingFace) |
| Manipulação de Dados | Pandas / NumPy                    |
| Interface            | React, JavaScript                 |

## Google colab

https://colab.research.google.com/drive/1AcidKOIanc64XdTVEW6PTRkf4c5L84Tw?usp=sharing#scrollTo=hcqBUpwTcX5A

## Vídeo Explicativo Youtube

https://www.youtube.com/watch?v=IuJX64d9ndQ

## 👤 Autores

Carlos Augusto Freire Maia de Oliveira
RA: 21.00781-0

Cesar Augusto Bresciani Junior
RA: 21.00478-0

Enzo Leonardo Sabatelli de Moura
RA: 21.01535-0
