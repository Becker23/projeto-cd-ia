import os
import re
import time
import pathlib
import requests
from typing import Optional, Tuple, List

# ====== CONFIG ======
WIKI_LANG = "pt"            # "pt" para Wikipédia em português
INTRO_ONLY = False          # True = só a introdução do artigo
SAVE_DIR = "saida_wiki"
MAX_CHARS_PROMPT = 12000
MODEL_PREF = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-pro",
]

# Limites de taxa (ajuste conforme seu plano)
FREE_TIER_RPM = 15               # requisições por minuto (ex.: free tier)
ENFORCE_CLIENT_THROTTLE = True   # aplica atraso mínimo entre chamadas
# ====================

# (opcional) .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import google.generativeai as genai

_last_call_ts = 0.0

def ensure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Defina GEMINI_API_KEY.\n"
            "PowerShell: setx GEMINI_API_KEY \"sua_chave\"  (abra novo terminal)\n"
            "ou na sessão atual: $env:GEMINI_API_KEY=\"sua_chave\""
        )
    genai.configure(api_key=api_key)

def pick_available_model(preferred: List[str]) -> str:
    """Lista modelos e escolhe o primeiro da preferência que suporta generateContent."""
    models = list(genai.list_models())
    def supports_generate(m):
        return hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods
    available = {m.name.replace("models/", "") for m in models if supports_generate(m)}
    for want in preferred:
        if want in available:
            return want
    if available:
        return sorted(list(available))[0]
    raise RuntimeError("Nenhum modelo com generateContent disponível para sua conta/versão.")

# ====== Wikipedia ======
WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WikipediaDataCollector/1.0 (+https://example.com/contact)"}

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:100]

def wiki_request(params: dict, lang: str, retries: int = 3, backoff: float = 1.5) -> dict:
    url = WIKI_API_URL.format(lang=lang)
    last_err = None
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 403:
                time.sleep(backoff ** (i + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))
    raise RuntimeError(f"Falha na Wikipedia após {retries} tentativas: {last_err}")

def search_wikipedia_title(query: str, lang: str) -> Optional[str]:
    params = {"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"}
    data = wiki_request(params, lang=lang)
    titles = data[1] if len(data) > 1 else []
    return titles[0] if titles else None

def get_wikipedia_article(title: str, lang: str = "en", intro_only: bool = False) -> Tuple[str, str]:
    params = {
        "action": "query",
        "format": "json",
        "redirects": 1,
        "converttitles": 1,
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
    }
    if intro_only:
        params["exintro"] = 1

    data = wiki_request(params, lang=lang)
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        alt = search_wikipedia_title(title, lang)
        if not alt:
            raise ValueError(f"Não encontrado na Wikipedia: '{title}'")
        return get_wikipedia_article(alt, lang=lang, intro_only=intro_only)

    page = next(iter(pages.values()))
    if page.get("missing") == "" or page.get("pageid") == -1:
        alt = search_wikipedia_title(title, lang)
        if not alt:
            raise ValueError(f"Página não encontrada: '{title}'")
        return get_wikipedia_article(alt, lang=lang, intro_only=intro_only)

    normalized_title = page.get("title", title)
    extract = page.get("extract", "")
    if not extract:
        raise ValueError(f"Artigo sem conteúdo legível: '{normalized_title}'")
    return normalized_title, extract

def truncate_for_prompt(text: str, max_chars: int = MAX_CHARS_PROMPT) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n\n[Texto truncado por comprimento.]"

# ====== Gemini com retry/espera ======
def _client_throttle_sleep():
    """Garante espaçamento mínimo entre chamadas para respeitar RPM do cliente."""
    global _last_call_ts
    if not ENFORCE_CLIENT_THROTTLE or FREE_TIER_RPM <= 0:
        return
    min_interval = 60.0 / float(FREE_TIER_RPM)
    now = time.time()
    elapsed = now - _last_call_ts
    if elapsed < min_interval:
        wait_s = min_interval - elapsed
        print(f"[throttle] Aguardando {wait_s:.2f}s para respeitar {FREE_TIER_RPM} rpm...")
        time.sleep(wait_s)
    _last_call_ts = time.time()

def _parse_retry_seconds_from_error(err_msg: str) -> Optional[float]:
    """
    Extrai 'Please retry in 36.281175467s' do texto do erro, se existir.
    """
    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", err_msg, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def call_gemini_with_retry(model, prompt: str, max_retries: int = 6):
    """
    Chama generate_content com:
      - throttle do cliente por RPM
      - retry automático se 429/quotas, usando retry_delay sugerido
      - backoff exponencial como fallback
    """
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            _client_throttle_sleep()
            resp = model.generate_content(prompt)
            if not resp or not getattr(resp, "text", None):
                raise RuntimeError("Resposta vazia do modelo.")
            return resp.text.strip()
        except Exception as e:
            msg = str(e)
            # Se vier "Please retry in Xs", usamos X
            retry_s = _parse_retry_seconds_from_error(msg)
            if retry_s is None:
                # fallback para backoff exponencial crescente (máx 60s)
                retry_s = min(backoff ** (attempt - 1), 60.0)
            print(f"[retry] Erro na chamada ({type(e).__name__}): {msg}\n"
                  f"→ Aguardando {retry_s:.2f}s e tentando novamente ({attempt}/{max_retries})...")
            time.sleep(retry_s)
    raise RuntimeError(f"Falhou após {max_retries} tentativas.")

def generate_text_with_gemini(base_text: str, model_name: str) -> str:
    prompt = (
        "Reescreva o texto abaixo de forma objetiva, neutra e enciclopédica,\n"
        "sem incluir títulos ou seções, sem linguagem promocional,\n"
        "sem estrutura de artigo didático.\n"
        "Estilo desejado: semelhante ao da Wikipédia.\n\n"
        "Texto de base:\n"
        f"{base_text}\n"
    )
    model = genai.GenerativeModel(model_name)
    try:
        text = call_gemini_with_retry(model, prompt, max_retries=6)
        return text
    except Exception:
        # tentativa alternativa
        alt = "Reformule de modo autoral e objetivo o texto a seguir, mantendo precisão:\n\n" + base_text
        return call_gemini_with_retry(model, alt, max_retries=6)

# ====== Arquivos ======
def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def save_texts(base_dir: str, title: str, original: str, generated: str) -> None:
    ensure_dir(base_dir)
    base = slugify(title)
    with open(os.path.join(base_dir, f"{base}__original.txt"), "w", encoding="utf-8") as f:
        f.write(original)
    with open(os.path.join(base_dir, f"{base}__ia.txt"), "w", encoding="utf-8") as f:
        f.write(generated)
    print(f"✓ Salvo: {base}__original.txt / {base}__ia.txt")

# ====== Pipeline ======
def collect_and_generate_for_title(raw_title: str, lang: str, intro_only: bool, model_name: str) -> None:
    print(f"Coletando e gerando para: {raw_title}")
    try:
        norm_title, original_text = get_wikipedia_article(raw_title, lang=lang, intro_only=intro_only)
        prompt_text = truncate_for_prompt(original_text)
        generated = generate_text_with_gemini(prompt_text, model_name=model_name)
        save_texts(SAVE_DIR, norm_title, original_text, generated)
    except Exception as e:
        print(f"✗ Falhou em '{raw_title}': {e}")

def main():
    ensure_gemini()
    model_name = pick_available_model(MODEL_PREF)
    print(f"Modelo Gemini selecionado: {model_name}")

    articles = [
        "Inteligência artificial",
        "Aprendizado de máquina",
        "Visão computacional",
        "Processamento de linguagem natural",
        "Robótica",
        "Computação quântica",
        "Blockchain",
        "Cibersegurança",
        "Criptografia",
        "Algoritmo genético",
        "Banco de dados",
        "Redes neurais artificiais",
        "Internet das coisas",
        "Sistemas embarcados",
        "Engenharia de software",
        "Computação em nuvem",
        "Big Data",
        "Ciência de dados",
        "Realidade virtual",
        "Realidade aumentada",
        "Estatística",
        "Probabilidade",
        "Álgebra linear",
        "Cálculo diferencial",
        "Cálculo integral",
        "Equações diferenciais",
        "Transformada de Fourier",
        "Teoria dos grafos",
        "Otimização matemática",
        "Geometria analítica",
        "Lógica matemática",
        "Análise combinatória",
        "Séries numéricas",
        "Inferência estatística",
        "Teoria da informação",
        "Mecânica clássica",
        "Termodinâmica",
        "Mecânica quântica",
        "Relatividade",
        "Eletromagnetismo",
        "Energia renovável",
        "Engenharia elétrica",
        "Engenharia mecânica",
        "Engenharia civil",
        "Engenharia de materiais",
        "Aerodinâmica",
        "Estruturas metálicas",
        "Máquinas térmicas",
        "Hidráulica",
        "Circuito elétrico",
        "Química orgânica",
        "Química inorgânica",
        "Tabela periódica",
        "Ligações químicas",
        "Reações químicas",
        "Catálise",
        "Proteína",
        "Aminoácido",
        "Lipídio",
        "Carboidrato",
        "Metabolismo",
        "Fotossíntese",
        "Respiração celular",
        "Soluções químicas",
        "Polímeros",
        "Biotecnologia",
        "Microbiologia",
        "DNA",
        "RNA",
        "Genética",
        "Célula-tronco",
        "Sistema nervoso",
        "Sistema imunológico",
        "Vacina",
        "Vírus",
        "Bactéria",
        "Hormônios",
        "Anatomia humana",
        "Bioinformática",
        "Evolução biológica",
        "Reprodução humana",
        "Hemoglobina",
        "Sistema cardiovascular",
        "Sistema respiratório",
        "Doenças infecciosas",
        "Revolução Francesa",
        "Iluminismo",
        "Idade Média",
        "Renascimento",
        "Revolução Industrial",
        "Guerra Fria",
        "Segunda Guerra Mundial",
        "Primeira Guerra Mundial",
        "Antigo Egito",
        "Grécia Antiga",
        "Roma Antiga",
        "Idade Moderna",
        "Escravidão no Brasil",
        "Ditadura militar no Brasil",
        "Independência do Brasil",
        "Revolução Russa",
        "Império Bizantino",
        "Mesopotâmia",
        "Incas",
        "Maias",
        "Economia",
        "Macroeconomia",
        "Microeconomia",
        "Inflação",
        "Mercado financeiro",
        "Capitalismo",
        "Socialismo",
        "Liberalismo",
        "Filosofia",
        "Ética",
        "Epistemologia",
        "Sociologia",
        "Antropologia",
        "Direito constitucional",
        "Política pública"
    ]
    print(f"Wikipedia lang: {WIKI_LANG} | INTRO_ONLY={INTRO_ONLY}")
    for t in articles:
        collect_and_generate_for_title(t, lang=WIKI_LANG, intro_only=INTRO_ONLY, model_name=model_name)

if __name__ == "__main__":
    main()
