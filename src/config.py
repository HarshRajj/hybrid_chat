import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Look for .env in the project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded configuration from {env_path}")
    else:
        print(f"ℹ No .env file found at {env_path} - using environment variables")
except ImportError:
    print("ℹ python-dotenv not installed - using environment variables only")


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    
    if required and not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Please set it in your .env file or environment."
        )
    
    return value or ""


# Neo4j Configuration
NEO4J_URI = get_env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD", required=True)

# OpenAI Configuration
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
EMBED_MODEL = get_env("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = get_env("CHAT_MODEL", "gpt-4o-mini")

# Cerebras Configuration (Optional - for ultra-fast inference)
CEREBRAS_API_KEY = get_env("CEREBRAS_API_KEY", default=None)
CEREBRAS_MODEL = get_env("CEREBRAS_MODEL", "llama3.1-8b")  # Options: llama3.1-8b, llama3.1-70b
USE_CEREBRAS = get_env("USE_CEREBRAS", "false").lower() == "true"

# Pinecone Configuration
PINECONE_API_KEY = get_env("PINECONE_API_KEY", required=True)
PINECONE_ENV = get_env("PINECONE_ENV", "us-east1-gcp")
PINECONE_INDEX_NAME = get_env("PINECONE_INDEX_NAME", "vietnam-travel")
PINECONE_VECTOR_DIM = int(get_env("PINECONE_VECTOR_DIM", "1536"))


def validate_config() -> bool:

    try:
        # Test all required variables
        get_env("NEO4J_PASSWORD", required=True)
        get_env("OPENAI_API_KEY", required=True)
        get_env("PINECONE_API_KEY", required=True)
        return True
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        raise


if __name__ == "__main__":
    """Print current configuration (for debugging)"""
    print("\n" + "="*60)
    print("Current Configuration")
    print("="*60)
    print(f"NEO4J_URI:           {NEO4J_URI}")
    print(f"NEO4J_USER:          {NEO4J_USER}")
    print(f"NEO4J_PASSWORD:      {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else '❌ NOT SET'}")
    print(f"OPENAI_API_KEY:      {'*' * min(20, len(OPENAI_API_KEY)) if OPENAI_API_KEY else '❌ NOT SET'}")
    print(f"CEREBRAS_API_KEY:    {'*' * min(20, len(CEREBRAS_API_KEY)) if CEREBRAS_API_KEY else 'Not configured (optional)'}")
    print(f"USE_CEREBRAS:        {USE_CEREBRAS}")
    print(f"CEREBRAS_MODEL:      {CEREBRAS_MODEL if USE_CEREBRAS else 'N/A'}")
    print(f"CHAT_MODEL:          {CHAT_MODEL if not USE_CEREBRAS else 'N/A (using Cerebras)'}")
    print(f"PINECONE_API_KEY:    {'*' * min(20, len(PINECONE_API_KEY)) if PINECONE_API_KEY else '❌ NOT SET'}")
    print(f"PINECONE_ENV:        {PINECONE_ENV}")
    print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")
    print(f"PINECONE_VECTOR_DIM: {PINECONE_VECTOR_DIM}")
    print("="*60)
    
    try:
        validate_config()
        print("✓ All required configuration is set")
    except ValueError:
        print("\n⚠ Please create a .env file with all required variables")
        print("See .env.example for a template")
    print()
