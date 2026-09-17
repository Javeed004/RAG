import json
from pathlib import Path
import os
import requests
import streamlit as st


# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# CONFIG_FILE = PROJECT_ROOT / "config.json"

# with open(CONFIG_FILE, "r", encoding="utf-8") as file:
#     config = json.load(file)

# API_URL = config["backend"]["base_url"]
API_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
)

SUPPORTED_TYPES = ["pdf", "txt", "docx", "csv", "xlsx"]
SUPPORTED_LABEL = "PDF, TXT, DOCX, CSV, XLSX"


# PAGE CONFIGURATION

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon=None,
    layout="centered",
)


# STYLING

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.5rem;
            max-width: 780px;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }
        .app-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .app-subtitle {
            color: var(--text-color-secondary, #6b7280);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        .source-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        }
        .source-card-header {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .source-meta {
            font-size: 0.8rem;
            color: var(--text-color-secondary, #6b7280);
            margin-bottom: 0.35rem;
        }
        .upload-status-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.15rem 0;
        }
        .status-ok {
            color: #15803d;
            font-weight: 600;
        }
        .status-fail {
            color: #b91c1c;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_sources(sources):
    """
    Display retrieved RAG chunks with metadata
    and similarity scores.
    """

    if not sources:
        st.caption("No sources were returned.")
        return

    with st.expander(f"View sources ({len(sources)})"):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            source_name = source.get(
                "source",
                "Unknown",
            )

            page = source.get(
                "page",
                "N/A",
            )

            similarity = source.get(
                "similarity",
            )

            distance = source.get(
                "distance",
            )

            chunk = source.get(
                "chunk",
                "",
            )

            meta_parts = [f"Document: {source_name}", f"Page: {page}"]

            if similarity is not None:
                meta_parts.append(f"Similarity: {similarity:.4f}")

            if distance is not None:
                meta_parts.append(f"Distance: {distance:.4f}")

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-card-header">Source {index}</div>
                    <div class="source-meta">{" &nbsp;|&nbsp; ".join(meta_parts)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.code(
                chunk,
                language=None,
            )

            if index < len(sources):
                st.divider()


def upload_single_file(uploaded_file):
    """
    Upload one file to the FastAPI /upload endpoint.
    Returns (success: bool, message: str).
    """
    try:
        response = requests.post(
            f"{API_URL}/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        return True, data.get("message", "Document added successfully.")

    except requests.exceptions.Timeout:
        return False, "Processing timed out."

    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the backend."

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get("detail", "Upload failed.")
        except Exception:
            detail = "Upload failed."
        return False, detail

    except Exception as error:
        return False, f"Upload failed: {error}"


# SIDEBAR: DOCUMENT MANAGEMENT

with st.sidebar:
    st.header("Document Management")
    st.caption(f"Supported formats: {SUPPORTED_LABEL}")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help=f"Accepted file types: {SUPPORTED_LABEL}",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

        if st.button("Add to Knowledge Base", use_container_width=True):

            results = []
            progress = st.progress(0, text="Starting upload...")

            for index, uploaded_file in enumerate(uploaded_files, start=1):

                progress.progress(
                    (index - 1) / len(uploaded_files),
                    text=f"Uploading {uploaded_file.name} "
                         f"({index}/{len(uploaded_files)})...",
                )

                success, message = upload_single_file(uploaded_file)
                results.append((uploaded_file.name, success, message))

            progress.progress(1.0, text="Done.")
            progress.empty()

            # SUMMARY

            succeeded = [r for r in results if r[1]]
            failed = [r for r in results if not r[1]]

            if succeeded:
                st.success(
                    f"{len(succeeded)}/{len(results)} file(s) added successfully."
                )

            if failed:
                st.error(
                    f"{len(failed)} file(s) failed to upload."
                )

            with st.expander("View details", expanded=bool(failed)):
                for filename, ok, message in results:
                    status_class = "status-ok" if ok else "status-fail"
                    status_text = "Success" if ok else "Failed"
                    st.markdown(
                        f"""
                        <div class="upload-status-row">
                            <span class="{status_class}">{status_text}</span>
                            <span><strong>{filename}</strong> — {message}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.divider()

    if st.session_state.get("messages"):
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# TITLE

st.markdown('<div class="app-title">RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions about your uploaded documents.</div>',
    unsafe_allow_html=True,
)


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []


# DISPLAY CHAT HISTORY

if not st.session_state.messages:
    st.info(
        "Upload documents from the sidebar, then ask a question to get started."
    )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            display_sources(
                message["sources"]
            )


# CHAT INPUT

question = st.chat_input("Ask a question about your documents...")


# PROCESS QUESTION

if question:
    # BUILD HISTORY IN BACKEND-COMPATIBLE SHAPE

    # Backend expects: list[{"role": str, "content": str}]
    # Our session messages already have this shape plus optional "sources".

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
    ]

    # DISPLAY USER MESSAGE

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # CALL FASTAPI /chat ENDPOINT

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "history": history,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                # Backend ChatResponse:
                # { "answer": str, "sources": list[str] }
                answer = data.get("answer", "I don't know.")
                sources = data.get("sources", [])

                # DISPLAY ANSWER

                st.markdown(answer)

                # DISPLAY SOURCES

                display_sources(sources)

                # SAVE SUCCESSFUL RESPONSE

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The backend took too long to respond. "
                    "Please try again."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the RAG backend. "
                    "Make sure FastAPI is running."
                )

            except requests.exceptions.HTTPError:
                try:
                    detail = response.json().get(
                        "detail",
                        "Backend returned an error.",
                    )
                except Exception:
                    detail = "Backend returned an error."

                st.error(f"Backend error: {detail}")

            except Exception as error:
                st.error(f"Unexpected error: {error}")