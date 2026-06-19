# app/models.py
from pydantic import BaseModel, Field, BeforeValidator, model_validator
from typing import Optional, Literal, Annotated
from datetime import datetime
from bson import ObjectId

PyObjectId = Annotated[str, BeforeValidator(lambda v: str(v) if ObjectId.is_valid(v) else ValueError("Invalid ObjectId"))]

class User(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    first_name: str
    last_name:Optional[str] = None
    phone_number:Optional[str] = None
    email: str
    address_line_1:Optional[str] = None
    address_line_2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pin_code: Optional[str] = None

    # Plan info
    plan: Optional[Literal["free", "pro"]] = "free"
    billing_type: Optional[Literal["monthly", "yearly"]] = "monthly"
    price: Optional[float] = 0.0
    due_date: Optional[datetime] = None
    
    # 💡 Gateway Payment Tracking details directly stored in the User Document
    payment_id: Optional[str] = None
    order_id: Optional[str] = None
    
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_pro_due_date(self) -> "User":
        if self.plan == "pro" and not self.due_date:
            raise ValueError("due_date is required when the plan is set to 'pro'")
        return self