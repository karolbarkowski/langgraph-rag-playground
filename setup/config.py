import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

__all__ = ['SetupConfig', 'setupConfig']

@dataclass(frozen=True)
class SetupConfig:
    """Application configuration loaded from environment variables"""
    MONGO_URI: str
    DATABASE_NAME: str
    DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME: str
    PRODUCTS_COLLECTION: str


setupConfig = SetupConfig(
    MONGO_URI=os.environ["MONGO_URI"],
    DATABASE_NAME=os.environ["DATABASE_NAME"],
    DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME=os.environ["DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME"],
    PRODUCTS_COLLECTION=os.environ["PRODUCTS_COLLECTION"],
)
