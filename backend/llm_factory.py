import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm(provider_name=None):
    """
    Create and return the configured LangChain chat model.

    Supported providers:
    - ollama
    - openai
    """

    provider = (
        provider_name
        or os.getenv("LLM_PROVIDER", "ollama")
    ).strip().lower()

    # OLLAMA

    if provider == "ollama":

        model_name = os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:3b",
        )

        try:
            return ChatOllama(
                model=model_name,
                temperature=0,
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize Ollama model "
                f"'{model_name}': {error}"
            ) from error

    # OPENAI

    if provider == "openai":

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Add it to your .env file."
            )

        model_name = os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1-mini",
        )

        try:
            return ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key,
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize OpenAI model "
                f"'{model_name}': {error}"
            ) from error
            
    if provider == "groq":

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Add it to your .env file."
            )

        model_name = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )

        try:
            return ChatGroq(
                model=model_name,
                temperature=0,
                api_key=api_key,
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize Groq model "
                f"'{model_name}': {error}"
            ) from error

    # INVALID PROVIDER

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        f"Supported providers are: ollama, openai."
    )
    
def get_llm_config():
    provider = os.getenv(
        "LLM_PROVIDER",
        "ollama",
    ).strip().lower()

    model_name = os.getenv(
        f"{provider.upper()}_MODEL",
        "unknown",
    )

    return provider, model_name