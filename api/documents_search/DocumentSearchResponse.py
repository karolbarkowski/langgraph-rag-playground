from pydantic import BaseModel, Field


from typing import Dict, List


class DocumentSearchResponse(BaseModel):
    llm_response: str = Field(..., alias="LlmResponse")
    documents: List[Dict] = Field(..., alias="Documents")
    query: str
    timestamp: str
    processing_time_ms: float

    class Config:
        populate_by_name = True