# RAG Python Supabase

Yksinkertainen RAG (Retrieval-Augmented Generation) ratkaisu Supabase + OpenAI:lla.

## 🚀 Pika-aloitus

1. **Asenna riippuvuudet:**

   ```bash
   uv sync
   ```

2. **Luo `.env` tiedosto:**

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_anon_key_here
   ```

3. **⚠️ Tärkeää - Aja Supabase SQL-editorissa:**

   ```bash
   # Kopioi ja aja src/database/table.sql sisältö Supabase SQL-editorissa
   ```

   Tämä luo tarvittavat taulut ja funktiot vektoritietokantaa varten.

4. **Käynnistä sovellus:**

   ```bash
   streamlit run streamlit_app.py
   ```

## 📝 Huomiot

- Hanki OpenAI API-avain: [platform.openai.com](https://platform.openai.com/api-keys)
- Luo Supabase projekti: [supabase.com](https://supabase.com)
- **Muista ajaa `table.sql` ennen ensimmäistä käyttöä!**

## 📝 Huomiot

- Hanki OpenAI API-avain: [platform.openai.com](https://platform.openai.com/api-keys)
- Luo Supabase projekti: [supabase.com](https://supabase.com)
- **Muista ajaa `table.sql` ennen ensimmäistä käyttöä!**
