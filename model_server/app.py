from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan():


    yield

app = FastAPI(lifespan=lifespan)

