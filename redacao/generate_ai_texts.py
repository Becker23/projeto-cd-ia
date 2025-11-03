import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# ==== CONFIG ====
load_dotenv()
ANO_ENEM=2018
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
OUTPUT_FILE = "ai_texts.parquet"
PROMPT = """
## 🧠 **PROMPT — Guia Definitivo para Criação de Redação ENEM (com textos motivadores)**

> **Instrução para a IA:**
> Você é uma inteligência artificial especializada em **redações do ENEM**.
> Sua tarefa é **produzir uma redação dissertativo-argumentativa** seguindo **rigorosamente** as normas e critérios do **Exame Nacional do Ensino Médio (ENEM)**, utilizando o **tema** e os **textos motivadores** que serão fornecidos pelo usuário.
>
> ---
>
> ### 🧩 1. Estrutura obrigatória
>
> A redação deve ter **entre 20 e 30 linhas**, organizada em **prosa dissertativo-argumentativa**, com:
>
> * **Introdução:** Apresentação do tema e da tese (ponto de vista a ser defendido).
> * **Desenvolvimento 1:** Primeiro argumento, fundamentado logicamente.
> * **Desenvolvimento 2:** Segundo argumento, complementando o anterior.
> * **Conclusão:** Retomada da tese e apresentação de uma **proposta de intervenção social detalhada**.
>
> O texto deve ser **coeso, coerente, objetivo e formal**, sem listas, diálogos ou estrutura narrativa.
>
> ---
>
> ### 📚 2. Uso obrigatório dos textos motivadores
>
> * Os **textos motivadores** fornecidos devem ser **lidos, compreendidos e utilizados** para embasar os argumentos.
> * É **obrigatório** incorporar ideias, dados ou reflexões extraídos deles, **sem cópia literal**.
> * O uso deve ser **crítico, interpretativo e integrado** à argumentação, demonstrando repertório sociocultural.
>
> ---
>
> ### 🧱 3. Regras fundamentais
>
> * Utilize **apenas a norma culta da língua portuguesa**.
> * O texto deve ser **inteiramente original** e **respeitoso aos direitos humanos**.
> * **A tese** precisa ser clara e defendida ao longo de todo o texto.
> * Use **conectivos e articuladores** adequados para garantir coesão e progressão lógica.
> * **Jamais fuja do tema** ou altere o tipo textual.
>
> ---
>
> ### 🎯 4. Competências do ENEM
>
> 1. **Domínio da norma culta** da língua portuguesa.
> 2. **Compreensão do tema** e adequação ao gênero dissertativo-argumentativo.
> 3. **Organização de argumentos** de modo coerente e coeso.
> 4. **Uso apropriado de recursos linguísticos** na argumentação.
> 5. **Proposta de intervenção** detalhada, viável e ética, respeitando os direitos humanos.
>
> ---
>
> ### 🧩 5. Estrutura da proposta de intervenção
>
> A conclusão deve conter uma **proposta de intervenção** com os cinco elementos obrigatórios:
>
> * **Agente:** quem executa a ação;
> * **Ação:** o que será feito;
> * **Meio/modo:** como será feito;
> * **Finalidade:** por que será feito (objetivo social);
> * **Detalhamento:** local, recursos, etapas ou consequências positivas.
>
> ---
>
> ### ⚖️ 6. Postura ética e adequação
>
> * **Proibido:** discurso de ódio, ironia, linguagem informal, plágio ou fuga ao tema.
> * **Obrigatório:** tom formal, postura crítica e respeito aos direitos humanos.
>
> ---
>
> ### 🧠 7. Forma de resposta esperada
>
> Ao receber o **tema** e os **textos motivadores**, siga estes passos:
>
> 1. Analise o tema e identifique o problema central.
> 2. Utilize os textos motivadores para embasar ideias e dados.
> 3. Elabore uma **tese clara**.
> 4. Desenvolva **dois parágrafos argumentativos**.
> 5. Conclua com uma **proposta de intervenção detalhada**.
>
> ---
>
> ### 🗒️ 8. Forma final de saída
>
> 🔴 **Muito importante:**
>
> * Sua resposta deve conter **apenas e somente o texto da redação completa**,
> * **Sem título**, **sem comentários**, **sem explicações**, **sem marcações**, **sem repetições do tema** e **sem qualquer texto fora da redação**.
> * O texto deve começar imediatamente com a introdução e terminar com a conclusão.
>
> ---
>
> ### 🧾 9. Exemplo de uso
>
> Quando o usuário enviar algo como:
>
> ```
> Tema: A influência da tecnologia nas relações humanas.  
>
> Textos motivadores:  
> [Texto 1] Pesquisas apontam que a comunicação digital tem substituído o contato presencial em diversos contextos sociais.  
> [Texto 2] Estudos psicológicos destacam o aumento do isolamento social em jovens conectados à internet.  
> [Texto 3] O avanço tecnológico trouxe facilidades, mas também desafios para a empatia e o convívio interpessoal.
> ```
>
> Você deverá gerar **apenas a redação completa**, conforme todas as instruções acima.
>
> ---
>
> **Fim do guia.**
> Aguarde o envio do *tema* e dos *textos motivadores* antes de redigir.
---
Tema: “Manipulação do comportamento do usuário pelo controle de dados na internet”
---
Texto 1:
Às segundas-feiras pela manhã, os usuários de um serviço de música digital recebem uma lista personalizada de músicas que lhes permite descobrir novidades. Assim como os sistemas de outros aplicativos e redes sociais, este cérebro artificial consegue traçar um retrato automatizado do gosto de seus assinantes e constrói uma máquina de sugestões que não costuma falhar. O sistema se baseia em um algoritmo cuja evolução e usos aplicados ao consumo cultural são infinitos. De fato, plataformas de transmissão de vídeo on-line começam a desenhar suas séries de sucesso rastreando o banco de dados gerado por todos os movimentos dos usuários para analisar o que os satisfaz. O algoritmo constrói assim um universo cultural adequado e complacente com o gosto do consumidor, que pode avançar até chegar sempre a lugares reconhecíveis. Dessa forma, a filtragem da informação feita pelas redes sociais ou pelos sistemas de busca pode moldar nossa maneira de pensar. E esse é o problema principal: a ilusão de liberdade de escolha que muitas vezes é gerada pelos algoritmos.

VERDÚ, Daniel. O gosto na era do algoritmo. Disponível em: https://brasil.elpais.com. Acesso em: 11 jun. 2018 (adaptado).
---
Texto 2:
Nos sistemas de gigantes da internet, a filtragem de dados é transferida para um exército de moderadores em empresas localizadas do Oriente Médio do Sul da Ásia, que têm um papel importante no controle daquilo que deve ser eliminado da rede social, a partir de sinalizações dos usuários. Mas a informação é então processada por um algoritmo, que tem a decisão final. Os algoritmos são literais. Em poucas palavras, são uma opinião embrulhada em código. E estamos caminhando para um estágio em que é a máquina que decide qual notícia deve ou não ser lida.

PEPE ESCOBAR. A silenciosa ditadura do algoritmo. Disponível em: http://outraspalavras.net. Acesso em: 5 jun. 2017 (adaptado).
---
Texto 3:
Resumo da Mensagem Principal

A imagem é um infográfico que detalha a utilização da internet no Brasil por diferentes faixas etárias e as principais finalidades de acesso. A mensagem central é que a internet possui uma alta taxa de penetração na população com 10 anos ou mais, especialmente entre os jovens. Além disso, a internet é usada majoritariamente para a comunicação instantânea (aplicativos de mensagem), mas o consumo de vídeos e conteúdo audiovisual e a comunicação por chamadas (voz/vídeo) também são atividades muito populares, superando o uso tradicional de e-mail.

Descrição Detalhada para Pessoas com Deficiência Visual

Esta é uma imagem retangular, um infográfico em preto e branco com estatísticas apresentadas em números grandes e acompanhadas por ícones ilustrativos. O conteúdo é dividido em duas seções principais: "Utilização da Internet" e "Finalidade do acesso à Internet (%)".

1. Utilização da Internet (Taxa de Penetração):

    O texto introdutório da seção afirma: "64,7% das pessoas de 10 anos ou mais de idade utilizaram a internet."

    Dados por Gênero:

        Homens: Um ícone de figura masculina e o número 63,8%.

        Mulheres: Um ícone de figura feminina e o número 65,5%. Nota-se uma ligeira maior taxa de utilização entre as mulheres.

    Dados por Faixa Etária:

        Jovens (18 a 24 anos): "Cerca de 85% dos jovens de 18 a 24 anos de idade... utilizaram a internet."

        Idosos (60 anos ou mais): "...e 25% das pessoas de 60 anos ou mais de idade utilizaram a internet."

2. Finalidade do acesso à Internet (%) (As Quatro Mais Comuns):

Esta seção lista, em ordem decrescente (do maior percentual para o menor), os principais usos da internet:

    1º Comunicação por Mensagem (Texto/Voz/Imagem):

        Percentual: 94,2

        Ícone: Uma bolha de chat com reticências.

        Descrição: "Enviar ou receber mensagens de texto, voz ou imagens por aplicativos diferentes de e-mail."

    2º Consumo de Vídeo/Audiovisual:

        Percentual: 76,4

        Ícone: Uma tela de vídeo ou televisão com o símbolo de play e uma tira de filme.

        Descrição: "Assistir a vídeos, inclusive programas, séries e filmes."

    3º Comunicação por Chamada (Voz/Vídeo):

        Percentual: 73,3

        Ícone: Uma câmera de vídeo com um símbolo de som (alto-falante ou orelha).

        Descrição: "Conversar por chamada de voz ou vídeo."

    4º Uso de E-mail:

        Percentual: 69,3

        Ícone: Um envelope com o símbolo arroba (@).

        Descrição: "Enviar ou receber e-mails (correio eletrônico)."
---
Texto 4:
Mudanças sutis nas informações às quais somos expostos podem transformar nosso comportamento. As redes têm selecionado as notícias sob títulos chamativos como “trending topics” ou critérios como “relevância”. Mas nós praticamente não sabemos como isso tudo é filtrado. Quanto mais informações relevantes tivermos nas pontas dos dedos, melhor equipados estamos para tomar decisões. No entanto, surgem algumas tensões fundamentais: entre a conveniência e a deliberação; entre o que o usuário deseja e o que é melhor para ele; entre a transparência e o lado comercial. Quanto mais os sistemas souberem sobre você em comparação ao que você sabe sobre eles, há mais riscos de suas escolhas se tornarem apenas uma série de reações a “cutucadas” invisíveis. O que está em jogo não é tanto a questão “homem versus máquina”, mas sim a disputa “decisão informada versus obediência influenciada”.

CHATFIELD, Tom. Como a internet influencia secretamente nossas escolhas. Disponível em: www.bbc.com. Acesso em: 3 jun. 2017 (adaptado).
"""

NUM_SAMPLES = 13
SLEEP_BETWEEN = 2  # seconds
# =================

def call_gemini(prompt: str, temperature: float=1.0) -> str:
    headers = {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature
        }
    }
    resp = requests.post(API_URL, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    # Adjust extraction logic if response schema differs
    # Here assume result["candidates"][0]["content"]["parts"][0]["text"]
    try:
        generated = result["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
        print("Unexpected response:", result)
        raise
    return generated.strip()

def main():
    if not API_KEY:
        raise ValueError("Missing environment variable GEMINI_API_KEY")

    # Load existing or create new
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"Loaded existing file {OUTPUT_FILE} ({len(df)} entries)")
    else:
        df = pd.DataFrame(columns=["ano_enem", "response", "label"])
        print(f"Creating new dataset {OUTPUT_FILE}")

    for i in range(NUM_SAMPLES):
        temp = 0.9 + (i / (NUM_SAMPLES - 1)) * (1.1 - 0.9)
        print(f"\nGenerating sample {i+1}/{NUM_SAMPLES} …")
        try:
            answer = call_gemini(PROMPT, temp)
            print("Received answer (length:", len(answer), "chars)")
        except Exception as e:
            print("Error during API call:", e)
            continue

        new_row = pd.DataFrame({
            "ano_enem": [ANO_ENEM],
            "response": [answer],
            "label": ["ai"]
        })
        df = pd.concat([df, new_row], ignore_index=True)

        # Save
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE} (total entries: {len(df)})")

        if i < NUM_SAMPLES - 1:
            time.sleep(SLEEP_BETWEEN)

    print("\nDone! Generated texts saved.")

if __name__ == "__main__":
    main()
