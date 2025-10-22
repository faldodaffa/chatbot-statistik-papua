# --- IMPORTS ---
import streamlit as st
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings # Embedding lokal
from dotenv import load_dotenv
import time

# --- KONFIGURASI AWAL ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key tidak ditemukan. Mohon periksa file .env Anda.")
    st.stop()

# Konfigurasi genai
try:
     genai.configure(api_key=GOOGLE_API_KEY)
     st.sidebar.success("API Key Google dikonfigurasi.")
except Exception as e:
     st.sidebar.warning(f"Konfigurasi 'genai' gagal: {e}. Melanjutkan...")

# --- NAMA FILE PDF & INDEX ---
PDF_FILE_PATH = "statistik-daerah-provinsi-papua-2025.pdf"
FAISS_INDEX_PATH = "faiss_index_bps_papua_multi" # Ganti nama index agar rebuild

# --- MODEL EMBEDDING BARU ---
# Gunakan model embedding multilingual
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- FUNGSI EMBEDDING ---
@st.cache_resource
def get_embeddings_model(model_name):
    """Memuat model embedding HuggingFace."""
    st.info(f"Memuat model embedding '{model_name}' (hanya sekali)...")
    try:
        # Menentukan device (opsional, coba otomatis)
        # model_kwargs = {'device': 'cpu'} # Paksa CPU jika ada masalah GPU
        # encode_kwargs = {'normalize_embeddings': True} # Normalisasi jika diperlukan FAISS
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name
            # model_kwargs=model_kwargs,
            # encode_kwargs=encode_kwargs
        )
        st.success("Model embedding berhasil dimuat.")
        return embeddings
    except Exception as e:
        st.error(f"Gagal memuat model embedding: {e}")
        return None

# Muat model embedding baru
embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME)
if embeddings_model is None:
    st.stop()

# --- FUNGSI VECTOR STORE ---
# Cache resource untuk vector store
# Hapus argumen show_spinner karena kita pakai spinner manual di pemanggilan
@st.cache_resource
def load_or_create_vector_store(pdf_path, index_path, _embeddings):
    """Memuat index FAISS jika ada, atau membuatnya dari PDF jika belum ada."""
    # Penting: Nama index_path harus unik jika model embedding berubah agar cache rebuild
    if os.path.exists(index_path):
        st.sidebar.info(f"Mencoba memuat index FAISS dari '{index_path}'...")
        try:
            # Pastikan embedding yang dipakai sama saat load
            vector_store = FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)
            st.sidebar.success("Vector store berhasil dimuat dari disk.")
            return vector_store
        except Exception as e:
            st.sidebar.error(f"Gagal memuat index: {e}. Membuat ulang...")
    else:
        st.sidebar.warning(f"Index FAISS di '{index_path}' tidak ditemukan. Membuat index baru dari PDF...")

    if not os.path.exists(pdf_path):
        st.error(f"File PDF '{pdf_path}' tidak ditemukan.")
        return None
    try:
        # Spinner manual agar lebih jelas
        with st.spinner(f"Memproses PDF '{pdf_path}' dan membuat index baru dengan embedding multilingual... (Proses ini bisa memakan waktu cukup lama saat pertama kali!)"):
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len) # Coba chunk lebih kecil
            docs = text_splitter.split_documents(pages)
            st.sidebar.info(f"PDF dipecah ({len(docs)} bag.). Membuat index FAISS...")
            # Pembuatan index bisa lama tergantung jumlah docs & kekuatan CPU/GPU
            vector_store = FAISS.from_documents(docs, _embeddings)
            vector_store.save_local(index_path) # Simpan index baru
            st.success("Vector store baru berhasil dibuat dan disimpan!")
            return vector_store
    except Exception as e:
        st.error(f"Gagal memproses PDF atau membuat vector store: {e}")
        return None

# --- BUAT ATAU MUAT VECTOR STORE DENGAN NAMA BARU ---
vector_store = load_or_create_vector_store(PDF_FILE_PATH, FAISS_INDEX_PATH, embeddings_model)

if vector_store is None:
    st.warning("Gagal menyiapkan vector store. Aplikasi tidak bisa berjalan.")
    st.stop()

# --- FUNGSI GET RESPONSE (LangChain + gemini-2.5-flash + Bahasa Indonesia + Debugging Konteks) ---
# Fungsi ini tetap sama, hanya nama model LLM yang dipastikan benar
def get_rag_response(user_query, _vector_store, _api_key):
    """Searches docs (k=5), builds prompt (Indonesian), generates response using LangChain + Gemini-2.5-Flash."""
    if not _vector_store:
        return "Maaf, vector store belum siap."
    try:
        st.sidebar.info("Mencari dokumen relevan (k=5)...")
        docs = _vector_store.similarity_search(user_query, k=5) # Ambil 5 dokumen

        st.sidebar.subheader("Konteks yang Ditemukan (Multilingual):")
        if docs:
            for i, doc in enumerate(docs):
                st.sidebar.text(f"--- Dokumen {i+1} ---")
                st.sidebar.caption(doc.page_content[:300] + "...")
        else:
            st.sidebar.warning("Tidak ada potongan dokumen relevan yang ditemukan.")

        if not docs or user_query.strip().lower() in ["halo", "hi", "hai", "selamat pagi", "selamat siang", "selamat malam", "terima kasih"]:
             # ... (penanganan sapaan sama) ...
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
        Jika jawaban eksplisit tidak ada tapi informasi terkait ada di konteks, coba rangkum informasi terkait tersebut.
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

        # Gunakan gemini-2.5-flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Pastikan nama model ini benar
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
        # ... (Error handling sama) ...
        st.error(f"Error saat menjalankan RAG Chain: {e}")
        error_message = str(e).lower()
        if "quota" in error_message or "429" in error_message:
             st.warning("Batas kuota Google API terlampaui.")
             time.sleep(2)
        elif "404" in error_message and ("model" in error_message or "not found" in error_message):
             st.warning(f"Model LLM ('gemini-2.5-flash') tidak ditemukan/didukung oleh API.")
             return "Maaf, terjadi masalah saat menghubungi model AI (Model tidak ditemukan)."
        return "Maaf, terjadi kesalahan saat menghubungi AI."

# --- UI STREAMLIT ---
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