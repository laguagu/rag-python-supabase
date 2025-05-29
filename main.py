"""
RAG (Retrieval-Augmented Generation) sovellus Supabase-tietokannalla ja OpenAI embeddingsillä
"""

import logging
import os

from dotenv import load_dotenv

from src.rag.rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_sample_data(rag_system: RAGSystem):
    """Setup some sample data for testing"""
    sample_texts = [
        {
            "text": """
            Suomi on Pohjoismainen valtio, joka sijaitsee Euroopan pohjoisosassa. 
            Suomen pääkaupunki on Helsinki, ja maan väkiluku on noin 5,5 miljoonaa. 
            Suomi on tunnettu metsistään, järvistään ja saunastaan. 
            Maa kuuluu Euroopan unioniin ja sen valuutta on euro.
            """,
            "metadata": {"topic": "Suomi", "category": "maantieto"},
        },
        {
            "text": """
            Tekoäly (AI) on tietojenkäsittelytieteen ala, joka pyrkii luomaan älykkäitä koneita. 
            Koneoppiminen on tekoälyn osa-alue, jossa koneet oppivat datasta ilman eksplisiittistä ohjelmointia. 
            Syväoppiminen käyttää neuroverkkoja jäljittelemään ihmisaivojen toimintaa. 
            GPT-mallit ovat esimerkki suurista kielimalleista, jotka voivat tuottaa ihmismäistä tekstiä.
            """,
            "metadata": {"topic": "tekoäly", "category": "teknologia"},
        },
        {
            "text": """
            Python on korkean tason ohjelmointikieli, joka on tunnettu yksinkertaisuudestaan ja luettavuudestaan. 
            Se on suosittu datatieteessä, koneoppimisessa ja web-kehityksessä. 
            Python tarjoaa laajan kirjaston ekosysteemin, mukaan lukien NumPy, Pandas ja TensorFlow. 
            Kieli tukee useita ohjelmointiparadigmoja, mukaan lukien olio-ohjelmointi ja funktionaalinen ohjelmointi.
            """,
            "metadata": {"topic": "python", "category": "ohjelmointi"},
        },
    ]

    logger.info("Lisätään esimerkkitietoja tietokantaan...")

    for item in sample_texts:
        success = rag_system.add_text_document(item["text"], item["metadata"])
        if success:
            logger.info(f"Lisättiin dokumentti aiheesta: {item['metadata']['topic']}")
        else:
            logger.error(
                f"Dokumentin lisääminen epäonnistui: {item['metadata']['topic']}"
            )


def interactive_chat(rag_system: RAGSystem):
    """Interactive chat with the RAG system"""
    print("\n🤖 RAG Chat-järjestelmä käynnistetty!")
    print("Voit kysyä kysymyksiä tietokannassa olevista dokumenteista.")
    print("Kirjoita 'lopeta' päättääksesi keskustelun.\n")

    thread_id = "interactive_session"

    while True:
        try:
            # Get user input
            query = input("🧑 Sinä: ").strip()

            if query.lower() in ["lopeta", "quit", "exit"]:
                print("👋 Nähdään taas!")
                break

            if not query:
                continue

            print("🔍 Haetaan tietoa...")

            # Get answer from RAG system
            result = rag_system.ask(query, thread_id)

            print(f"\n🤖 Vastaus: {result['answer']}\n")

            # Show retrieved documents count
            doc_count = len(result["retrieved_docs"])
            if doc_count > 0:
                print(f"📚 Löytyi {doc_count} asiaan liittyvää dokumenttia.\n")

        except KeyboardInterrupt:
            print("\n👋 Keskustelu keskeytetty. Nähdään taas!")
            break
        except Exception as e:
            logger.error(f"Virhe keskustelussa: {e}")
            print("❌ Tapahtui virhe. Yritä uudelleen.\n")


def main():
    """Main application entry point"""
    print("🚀 Käynnistetään RAG-järjestelmä...")

    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Puuttuvat ympäristömuuttujat: {', '.join(missing_vars)}")
        print("📝 Kopioi .env.example tiedosto .env:ksi ja täytä tarvittavat arvot.")
        return

    try:
        # Initialize RAG system
        rag_system = RAGSystem()

        # Setup database - tables should be created manually in Supabase dashboard
        print("📊 Alustetaan tietokanta...")
        # Note: Use the SQL commands in src/database/table.sql to create tables manually

        # Add sample data
        setup_sample_data(rag_system)

        # Start interactive chat
        interactive_chat(rag_system)

    except Exception as e:
        logger.error(f"Sovelluksen käynnistys epäonnistui: {e}")
        print(f"❌ Virhe: {e}")
        print("🔧 Tarkista ympäristömuuttujat ja yhteys Supabaseen.")


if __name__ == "__main__":
    main()
