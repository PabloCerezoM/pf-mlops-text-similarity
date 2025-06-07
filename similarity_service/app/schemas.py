from pydantic import BaseModel

class TextPair(BaseModel):
    sentence1: str
    sentence2: str
