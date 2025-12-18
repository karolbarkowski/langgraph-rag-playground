# Pydantic models for request/response validation
from pydantic import BaseModel, Field


class ProductsSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User search query")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "blue Nike running shoes under $100"
            }
        }