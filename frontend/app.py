import requests
import streamlit as st


# CONFIGURATION

API_URL = "http://127.0.0.1:8000"


# PAGE CONFIGURATION

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="centered",
)

# SIDEBAR: DOCUMENT MANAGEMENT

with st.sidebar:
    st.header("Document Management")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
    )

    if uploaded_file is not None:
        if st.button("Add to Knowledge Base"):
            try:
                with st.spinner("Processing document..."):
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

                st.success(
                    data.get("message", "Document added successfully.")
                )

            except requests.exceptions.Timeout:
                st.error("Document processing timed out.")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend.")

            except requests.exceptions.HTTPError:
                try:
                    detail = response.json().get("detail", "Upload failed.")
                except Exception:
                    detail = "Upload failed."

                st.error(detail)

            except Exception as error:
                st.error(f"Upload failed: {error}")


# TITLE

st.title("📚 RAG Chatbot")
st.caption("Ask questions about your uploaded documents.")


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []


# DISPLAY CHAT HISTORY

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")


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

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.caption(f"📄 {source}")
                else:
                    st.caption("No sources were returned.")

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