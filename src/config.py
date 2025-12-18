import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

__all__ = ['LanggraphConfig', 'langgraphConfig']

@dataclass(frozen=True)
class LanggraphConfig:
    MONGO_URI: str
    DATABASE_NAME: str
    DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME: str
    PRODUCTS_COLLECTION: str
    PRODUCTS_VECTOR_INDEX_NAME: str
    DOCUMENTS_VECTOR_INDEX_NAME: str


# Create singleton config instance
langgraphConfig = LanggraphConfig(
    MONGO_URI=os.environ["MONGO_URI"],
    DATABASE_NAME=os.environ["DATABASE_NAME"],
    DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME=os.environ["DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME"],
    PRODUCTS_COLLECTION=os.environ["PRODUCTS_COLLECTION"],
    PRODUCTS_VECTOR_INDEX_NAME=os.environ["PRODUCTS_VECTOR_INDEX_NAME"],
    DOCUMENTS_VECTOR_INDEX_NAME=os.environ["DOCUMENTS_VECTOR_INDEX_NAME"],
)
