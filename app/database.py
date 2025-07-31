from motor.motor_asyncio import AsyncIOMotorClient
from decouple import config
import logging

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

db = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    try:
        db.client = AsyncIOMotorClient(config("MONGODB_URI"))
        db.database = db.client.get_database("video-variants")
        
        # Test the connection
        await db.client.admin.command('ping')
        logging.info("Successfully connected to MongoDB")
        
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logging.info("Disconnected from MongoDB")

def get_database():
    """Get database instance"""
    return db.database
