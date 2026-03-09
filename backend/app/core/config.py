from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuChat AI"
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    REDIS_URL: str = "redis://localhost:6379/0"
    MAX_UPLOAD_FILES: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()
