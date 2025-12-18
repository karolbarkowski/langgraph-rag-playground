from pydantic import BaseModel, Field


from typing import Dict, List


class ProductsSearchResponse(BaseModel):
    llm_response: str = Field(..., alias="LlmResponse")
    products: List[Dict] = Field(..., alias="Products")
    query: str
    timestamp: str
    processing_time_ms: float

    class Config:
        populate_by_name = True