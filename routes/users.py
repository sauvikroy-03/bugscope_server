# app/routes/users.py
import os
import razorpay
from fastapi import APIRouter, HTTPException, status
from database.db import db
from modals import userModal
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()  # safety net in case main.py didn't load it first


router = APIRouter(
    prefix="/api/users",
    tags=["Users"]
)

# Initialize the Razorpay Client with environment variables
# Fallback to dummy values to prevent app crashes if keys are temporarily missing


# Validation schema for incoming order creation requests
class OrderRequest(BaseModel):
    amount: int  # Amount in paise (e.g., ₹100 = 10000 paise)
    currency: str = "INR"


# ── 1. RAZORPAY ORDER GENERATION ENDPOINT (POST) ─────────────────────────────

@router.post("/create-order", status_code=status.HTTP_200_OK)
async def create_razorpay_order(payload: OrderRequest):
    """
    Generates a secure, verified Order tracking ID from Razorpay's servers.
    """
    try:
        razorpay_client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID", "YOUR_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET", "YOUR_SECRET")))
     
        order_data = {
            "amount": payload.amount,      # Amount in paise
            "currency": payload.currency,
            "payment_capture": 1           # Auto-capture payments instantly
        }
        
        # Request order token from Razorpay API
        razorpay_order = razorpay_client.order.create(data=order_data)
        return razorpay_order
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Razorpay Order Generation Failed: {str(e)}"
        )


# ── 2. INSERTION ENDPOINT (POST) ──────────────────────────────────────────────

@router.post(
    "", 
    status_code=status.HTTP_201_CREATED,
    response_model=userModal.User
)
async def create_user(payload: userModal.User):
    """
    Accepts a user payload, ensures email uniqueness, 
    inserts it into MongoDB using Motor, and returns the saved user.
    """
    # Check for duplicate email registration
    existing_user = await db.users.find_one({"email": payload.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )

    # Convert Pydantic model to dictionary with MongoDB custom _id naming mapping
    user_dict = payload.model_dump(by_alias=True)
    if user_dict.get("_id") is None:
        user_dict.pop("_id", None)

    # Insert into collection asynchronously
    result = await db.users.insert_one(user_dict)

    # Fetch and return the newly saved entity profile
    inserted_user = await db.users.find_one({"_id": result.inserted_id})
    return inserted_user


# ── 3. RETRIEVAL ENDPOINT - ALL USERS (GET) ───────────────────────────────────

@router.get(
    "", 
    response_model=List[userModal.User],
    status_code=status.HTTP_200_OK
)
async def get_all_users():
    """
    Fetches all user documents from the MongoDB 'users' collection.
    """
    users_list = []
    
    # Iterate through async pointer cursor to avoid thread blocks
    async for user_doc in db.users.find():
        users_list.append(user_doc)
        
    return users_list


# ── 4. RETRIEVAL ENDPOINT - SINGLE USER BY EMAIL (GET) ───────────────────────────

@router.get(
    "/email/{email}", 
    response_model=userModal.User,
    status_code=status.HTTP_200_OK
)
async def get_user_by_email(email: str):
    """
    Fetches a specific user document by their email string.
    """
    # Look up the user by email string directly
    user_doc = await db.users.find_one({"email": email})
    
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email '{email}' not found."
        )
        
    return user_doc