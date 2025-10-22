# --- IMPORTS ---
import streamlit as st
import google.generativeai as genai
import os
# Ganti import FAISS ke Chroma
from langchain_community.vectorstores import Chroma # GANTI IMPORT
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import shutil # Untuk menghapus folder index lama jika perlu

# --- KONFIGURASI AWAL ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key tidak ditemukan.")
    st.stop()

try:
     genai.configure(api_key=GOOGLE_API_KEY)
     st.sidebar.success("API Key Google dikonfigurasi.")
except Exception as e:
     st.sidebar.warning(f"Konfigurasi 'genai' gagal: {e}. Melanjutkan...")

# --- NAMA FILE PDF & PATH INDEX BARU ---
PDF_FILE_PATH = "statistik-daerah-provinsi-papua-2025.pdf"
CHROMA_DB_PATH = "./chroma_db_bps_papua" # Nama folder untuk ChromaDB

# --- MODEL EMBEDDING (Tetap sama) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

@st.cache_resource
def get_embeddings_model(model_name):
    st.info(f"Memuat model embedding '{model_name}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        st.success("Model embedding berhasil dimuat.")
        return embeddings
    except Exception as e:
        st.error(f"Gagal memuat model embedding: {e}")
        return None

embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME)
if embeddings_model is None:
    st.stop()

# --- FUNGSI VECTOR STORE (Diubah ke ChromaDB) ---
# Cache resource untuk vector store Chroma
@st.cache_resource(show_spinner=False)
def load_or_create_chroma_store(pdf_path, db_path, _embeddings):
    """Membuat index ChromaDB dari PDF jika belum ada."""
    # Chroma biasanya lebih baik dibuat ulang jika ada perubahan PDF/embedding
    # Untuk simplifikasi, kita buat ulang jika folder index belum ada
    if not os.path.exists(db_path):
        st.sidebar.warning(f"Index ChromaDB di '{db_path}' tidak ditemukan. Membuat index baru dari PDF...")

        if not os.path.exists(pdf_path):
            st.error(f"File PDF '{pdf_path}' tidak ditemukan.")
            return None
        try:
            with st.spinner(f"Memproses PDF '{pdf_path}' dan membuat index ChromaDB... (Bisa perlu waktu lama!)"):
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len)
                docs = text_splitter.split_documents(pages)
                st.sidebar.info(f"PDF dipecah ({len(docs)} bag.). Membuat index ChromaDB...")

                # Membuat ChromaDB dari dokumen. Ini akan otomatis menyimpan ke db_path jika persist_directory diset
                vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=_embeddings,
                    persist_directory=db_path # Menyimpan index ke disk
                )
                st.success("Vector store ChromaDB baru berhasil dibuat dan disimpan!")
                return vector_store
        except Exception as e:
            st.error(f"Gagal memproses PDF atau membuat vector store ChromaDB: {e}")
            return None
    else:
        # Jika folder index sudah ada, kita load saja
        st.sidebar.info(f"Mencoba memuat index ChromaDB dari '{db_path}'...")
        try:
             vector_store = Chroma(
                 persist_directory=db_path,
                 embedding_function=_embeddings
             )
             st.sidebar.success("Vector store ChromaDB berhasil dimuat dari disk.")
             return vector_store
        except Exception as e:
             st.sidebar.error(f"Gagal memuat index ChromaDB: {e}. Coba hapus folder '{db_path}' dan refresh.")
             return None # Gagal load


# --- BUAT ATAU MUAT VECTOR STORE CHROMA ---
vector_store = load_or_create_chroma_store(PDF_FILE_PATH, CHROMA_DB_PATH, embeddings_model)

if vector_store is None:
    st.warning("Gagal menyiapkan vector store ChromaDB.")
    st.stop()

# --- FUNGSI GET RESPONSE (Tetap sama, hanya LLM model dipastikan) ---
def get_rag_response(user_query, _vector_store, _api_key):
    """Searches docs (k=4), builds prompt (Indonesian), generates response using LangChain + Gemini-2.5-Flash."""
    # (Isi fungsi ini SAMA PERSIS seperti versi sebelumnya yang berhasil menjawab,
    #  pastikan model LLM adalah "gemini-2.5-flash" dan k=4 atau k=5)
    if not _vector_store:
        return "Maaf, vector store belum siap."
    try:
        st.sidebar.info("Mencari dokumen relevan (k=5)...") # Coba k=5
        docs = _vector_store.similarity_search(user_query, k=5)

        st.sidebar.subheader("Konteks yang Ditemukan (Chroma):")
        if docs:
            for i, doc in enumerate(docs):
                st.sidebar.text(f"--- Dokumen {i+1} ---")
                st.sidebar.caption(doc.page_content[:300] + "...")
        else:
            st.sidebar.warning("Tidak ada potongan dokumen relevan yang ditemukan.")

        if not docs or user_query.strip().lower() in ["halo", "hi", "hai", "selamat pagi", "selamat siang", "selamat malam", "terima kasih"]:
             if user_query.strip().lower() in ["terima kasih", "makasih", "ok", "oke"]:
                  return "Sama-sama! Ada lagi yang bisa saya bantu?"
             else:
                  return "Halo! Ada yang bisa saya bantu terkait Statistik Daerah Papua 2025?"
        st.sidebar.success(f"{len(docs)} dokumen relevan ditemukan.")

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt_template = """
        Anda adalah asisten AI yang menjawab pertanyaan berdasarkan dokumen Statistik Daerah Provinsi Papua 2025.
        Gunakan informasi dari konteks yang diberikan di bawah ini sebagai prioritas utama.
        Cobalah untuk menyimpulkan jawaban dari konteks jika memungkinkan.
        Jika pertanyaan sama sekali tidak dapat dijawab dari konteks, katakan bahwa Anda tidak menemukan jawabannya di dokumen.
        Jangan mengarang informasi di luar konteks yang diberikan.
        **JAWAB SELALU DALAM BAHASA INDONESIA.**

        Konteks:
        {context}

        Pertanyaan:
        {question}

        Jawaban (dalam Bahasa Indonesia, berdasarkan konteks):
        """
        prompt_text = prompt_template.format(context=context, question=user_query)
        st.sidebar.text("Prompt RAG dikirim ke LLM (potongan):")
        st.sidebar.code(prompt_text[:500] + "...")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Pastikan pakai model ini
            google_api_key=_api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        st.sidebar.info("Membuat QA Chain...")

        chain = load_qa_chain(llm, chain_type="stuff")
        st.sidebar.info("Menjalankan QA Chain dengan Gemini-2.5-Flash...")

        response = chain.invoke({"input_documents": docs, "question": user_query})
        st.sidebar.success("Respons diterima.")

        return response.get('output_text', "Gagal mendapatkan output teks.")

    except Exception as e:
        st.error(f"Error saat menjalankan RAG Chain: {e}")
        error_message = str(e).lower()
        if "quota" in error_message or "429" in error_message:
             st.warning("Batas kuota Google API terlampaui.")
             time.sleep(2)
        elif "404" in error_message and ("model" in error_message or "not found" in error_message):
             st.warning(f"Model LLM ('gemini-2.5-flash') tidak ditemukan/didukung oleh API.")
             return "Maaf, terjadi masalah saat menghubungi model AI (Model tidak ditemukan)."
        return "Maaf, terjadi kesalahan saat menghubungi AI."


# --- UI STREAMLIT (Sama seperti sebelumnya) ---
st.set_page_config(page_title="Chatbot Statistik Papua", page_icon="📊")
st.title("📊 Chatbot Statistik Daerah Provinsi Papua 2025")
st.caption("Tanya jawab berdasarkan dokumen 'Statistik Daerah Provinsi Papua 2025' (Model: Gemini 2.5 Flash)")
st.sidebar.header("Status Proses")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait Statistik Daerah Papua 2025? (Jawab dalam Bahasa Indonesia)"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Masukkan pertanyaan Anda (Bahasa Indonesia)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Memproses dengan LangChain + Gemini Flash..."):
                response_text = get_rag_response(prompt, vector_store, GOOGLE_API_KEY)
                st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        st.error("Vector store belum siap.")