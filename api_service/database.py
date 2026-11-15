import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Boolean, BigInteger, DateTime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://admin:secret@db:5432/corp_bot")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True)
    telegram_id = Column(BigInteger, unique=True, nullable=True)
    full_name = Column(String, default="Сотрудник")
    username = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    approved_by = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session