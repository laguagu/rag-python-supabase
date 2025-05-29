# RAG Python Supabase

Tämä on kokonainen RAG (Retrieval-Augmented Generation) ratkaisu, joka käyttää Supabasea vektoritietokantana ja OpenAI embeddingsiä. Projekti hyödyntää LangChain/LangGraph työkaluja ja on rakennettu Python-kielellä uv paketinhallintaa käyttäen.

## 🌟 Ominaisuudet

- **🔍 Vektoritietokantatuki**: Supabase + pgvector vektoritietokannan tallennukseen
- **🤖 OpenAI Embeddings**: Tekstin vektorointi OpenAI:n embedding-malleilla
- **🧠 LangGraph orkestrointi**: Älykkäs työnkulku retrieval + generation toimintojen koordinointiin
- **💬 Streamlit UI**: Kaunis web-käyttöliittymä chattaamiseen
- **📁 Dokumenttien lataus**: Tuki tekstitiedostojen lataamiseen
- **🔧 Kehittäjäystävällinen**: Modulaarinen arkkitehtuuri ja kattava loggaus

## 🚀 Aloittaminen

### 1. Riippuvuuksien asennus

```bash
# Asenna riippuvuudet uv:llä
uv sync
```

### 2. Ympäristömuuttujien asetus

Kopioi `.env.example` tiedosto `.env`:ksi ja täytä tarvittavat arvot:

```bash
cp .env.example .env
```

Muokkaa `.env` tiedostoa:

```env
# OpenAI API configuration
OPENAI_API_KEY=your_openai_api_key_here

# Supabase configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

### 3. Supabase asetus

1. Luo uusi projekti [Supabase.com](https://supabase.com)
2. Ota käyttöön pgvector-extensio SQL-editorissa:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Kopioi projektin URL ja anon key `.env` tiedostoon

### 4. OpenAI API-avain

1. Luo API-avain [OpenAI Platform](https://platform.openai.com/api-keys)
2. Lisää avain `.env` tiedostoon

## 📖 Käyttö

### Komentorivikäyttö

Käynnistä interaktiivinen chat:

```bash
python main.py
```

### Web-käyttöliittymä

Käynnistä Streamlit-sovellus:

```bash
streamlit run streamlit_app.py
```

### Dokumenttien lataus

Lataa yksittäinen tiedosto:
```bash
python document_loader.py --file polku/tiedostoon.txt
```

Lataa kaikki tiedostot hakemistosta:
```bash
python document_loader.py --directory polku/hakemistoon
```

Lataa teksti suoraan:
```bash
python document_loader.py --text "Tämä on esimerkkiteksti" --metadata '{"category": "example"}'
```

## 🏗️ Projektirakenne

```
rag-python-supabase/
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   └── supabase_manager.py    # Supabase yhteys ja vektorioperaatiot
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_manager.py   # OpenAI embeddings ja tekstinkäsittely
│   └── rag/
│       ├── __init__.py
│       └── rag_system.py          # LangGraph RAG-orkestrointi
├── data/                          # Lokaalit datatiedostot
├── main.py                        # Komentorivikäyttöliittymä
├── streamlit_app.py              # Web-käyttöliittymä
├── document_loader.py            # Dokumenttien lataustyökalu
├── pyproject.toml               # Riippuvuudet
├── .env.example                 # Ympäristömuuttujien malli
└── README.md                    # Tämä tiedosto
```

## 🔧 Komponentin kuvaus

### RAGSystem (LangGraph)

Ydin-RAG järjestelmä, joka käyttää LangGraphia työnkulun orchestrointiin:

- **Retrieve-vaihe**: Hakee asiaan liittyvät dokumentit Supabasesta
- **Generate-vaihe**: Generoi vastauksen OpenAI:n avulla
- **Memory**: Säilyttää keskustelun kontekstin

### SupabaseManager

Hallitsee yhteyttä Supabase vektoritietokantaan:

- Automaattinen taulujen luonti
- Vektorihaut pgvector-extensiolla
- Dokumenttien lisäys ja päivitys

### EmbeddingManager

Käsittelee tekstin vektorointia ja dokumenttien jakamista:

- OpenAI embedding-mallit
- Tekstin jakaminen sopiviin osiin
- Token-laskenta ja optimointi

## 🛠️ Kehitysympäristö

### Riippuvuuksien hallinta

Projekti käyttää [uv](https://docs.astral.sh/uv/) paketinhallintaa:

```bash
# Lisää uusi riippuvuus
uv add package-name

# Päivitä riippuvuudet
uv sync

# Aja komentoja virtuaaliympäristössä
uv run python main.py
```

### Loggaus

Järjestelmä käyttää Python logging-moduulia. Lokien taso voidaan muuttaa koodissa:

```python
logging.basicConfig(level=logging.DEBUG)  # Yksityiskohtaiset logit
```

## 🤝 Vianmääritys

### Yleiset ongelmat

1. **"Import virheet"**
   - Varmista että uv sync on ajettu
   - Tarkista että olet oikeassa hakemistossa

2. **"Supabase yhteys epäonnistuu"**
   - Tarkista SUPABASE_URL ja SUPABASE_KEY
   - Varmista että pgvector-extensio on käytössä

3. **"OpenAI API virheet"**
   - Tarkista OPENAI_API_KEY
   - Varmista että sinulla on riittävästi krediittejä

4. **"Embedding virheet"**
   - Tarkista tekstin koko (max ~8000 tokenia per chunk)
   - Varmista että teksti on oikeassa koodauksessa (UTF-8)

### Debug-tila

Käynnistä yksityiskohtaisella logituksella:

```python
# main.py alussa
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 Lisenssi

MIT License - vapaasti käytettävissä

## 🤝 Kehittäminen

Tervetuloa osallistumaan projektin kehittämiseen! 

1. Forkkaa repositorio
2. Luo feature-branch
3. Tee muutokset
4. Testaa huolellisesti
5. Luo pull request

## 🆘 Tuki

Jos tarvitset apua:

1. Tarkista tämä README.md
2. Katso virhelokeja
3. Tarkista ympäristömuuttujat
4. Luo issue GitHubissa