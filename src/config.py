from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  google_api_key: str
  chroma_api_key: str
  chroma_tenant: str
  chroma_database: str
  chroma_collection: str = "fireflies_transcripts"

  model_config = SettingsConfigDict(
      env_file=".env", env_file_encoding="utf-8", extra="ignore"
  )


# Instantiate once so it can be imported globally
settings = Settings()