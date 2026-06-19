# app/db.py
import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv()


load_dotenv()

# Replace with your actual connection string via environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())

db = client["bugscope"]