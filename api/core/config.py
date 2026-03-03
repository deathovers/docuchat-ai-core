import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuChat AI"
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_ENVIRONMENT: str = "us-east-1"
    
    class Config:
        env_file = ".env"

settings = Settings()
