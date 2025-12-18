from pydantic import BaseModel, Field


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question about documents")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the return policy for shoes?"
            }
        }