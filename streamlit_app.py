"""
Streamlit web-käyttöliittymä RAG-järjestelmälle
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG system
try:
    from src.rag.rag_system import RAGSystem
except ImportError as e:
    st.error(f"Virhe RAG-järjestelmän tuonnissa: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG Chat-järjestelmä",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    try:
        # Check environment variables
        required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            st.error(f"❌ Puuttuvat ympäristömuuttujat: {', '.join(missing_vars)}")
            st.info(
                "📝 Kopioi .env.example tiedosto .env:ksi ja täytä tarvittavat arvot."
            )
            return None
        # Initialize RAG system
        rag_system = RAGSystem()

        # Note: Tables should be created manually in Supabase dashboard using table.sql
        # rag_system.supabase_manager.create_tables_if_not_exists()  # Removed - use manual setup

        return rag_system

    except Exception as e:
        st.error(f"Virhe RAG-järjestelmän alustuksessa: {e}")
        return None


def add_sample_data(rag_system):
    """Add sample data to the system"""
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

    for item in sample_texts:
        rag_system.add_text_document(item["text"], item["metadata"])


def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 RAG Chat-järjestelmä</h1>
        <p>Kysele tietoja tietokannasta käyttäen OpenAI embeddingsiä ja Supabase vektoritietokantaa</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize RAG system
    rag_system = initialize_rag_system()

    if not rag_system:
        st.warning(
            "⚠️ RAG-järjestelmä ei ole käytettävissä. Tarkista ympäristömuuttujat."
        )
        return

    # Sidebar
    st.sidebar.markdown(
        """
    <div class="sidebar-content">
        <h3>🔧 Asetukset</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Add sample data button
    if st.sidebar.button("📚 Lisää esimerkkitiedot"):
        with st.spinner("Lisätään esimerkkitietoja..."):
            add_sample_data(rag_system)
            st.sidebar.success("✅ Esimerkkitiedot lisätty!")

    # File upload
    st.sidebar.markdown("### 📁 Lisää tiedostoja")
    uploaded_file = st.sidebar.file_uploader(
        "Valitse tekstitiedosto",
        type=["txt"],
        help="Voit ladata .txt tiedostoja tietokantaan",
    )

    if uploaded_file and st.sidebar.button("📤 Lataa tiedosto"):
        try:
            # Save uploaded file temporarily
            temp_path = f"data/{uploaded_file.name}"
            os.makedirs("data", exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Add to RAG system
            with st.spinner("Käsitellään tiedostoa..."):
                success = rag_system.add_documents_from_files([temp_path])

            if success:
                st.sidebar.success(f"✅ Tiedosto {uploaded_file.name} lisätty!")
            else:
                st.sidebar.error("❌ Tiedoston lisääminen epäonnistui")

            # Clean up temp file
            os.remove(temp_path)

        except Exception as e:
            st.sidebar.error(f"Virhe tiedoston käsittelyssä: {e}")

    # Main chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Keskustelu")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Kysy jotain..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from RAG system
            with st.chat_message("assistant"):
                with st.spinner("Mietin vastausta..."):
                    try:
                        result = rag_system.ask(prompt, thread_id="streamlit_session")
                        response = result["answer"]

                        st.markdown(response)

                        # Store assistant response
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                        # Show retrieved documents in sidebar
                        with col2:
                            if result["retrieved_docs"]:
                                st.markdown("### 📖 Löydetyt dokumentit")
                                for i, doc in enumerate(result["retrieved_docs"], 1):
                                    with st.expander(f"Dokumentti {i}"):
                                        st.write(doc.page_content[:200] + "...")
                                        if doc.metadata:
                                            st.json(doc.metadata)

                    except Exception as e:
                        error_msg = f"Virhe vastausta generoitaessa: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

    with col2:
        st.markdown("### ℹ️ Tietoja")
        st.info("""
        Tämä on RAG (Retrieval-Augmented Generation) -sovellus, joka:
        
        🔍 **Hakee** asiaan liittyviä dokumentteja tietokannasta
        
        🤖 **Generoi** vastauksia OpenAI:n avulla
        
        💾 **Tallentaa** dokumentit Supabase vektoritietokantaan
        
        🧠 **Muistaa** keskustelun kontekstin
        """)

        # Clear chat button
        if st.button("🗑️ Tyhjennä keskustelu"):
            st.session_state.messages = []
            st.rerun()

        # Show system status        st.markdown("### 🔧 Järjestelmän tila")
        try:
            # Simple health check
            rag_system.ask("testi", thread_id="health_check")
            st.success("🟢 Järjestelmä toimii")
        except Exception as e:
            st.error(f"🔴 Järjestelmässä on ongelma: {e}")


if __name__ == "__main__":
    main()
