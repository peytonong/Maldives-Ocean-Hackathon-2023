from fastapi import Body, FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from os import environ as env
from routers import test_route
# from routers import post, auth, user, pet, quest, dialog, chat
# from database import database


app = FastAPI()

origins = [
  '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(test_route.router)
# app.include_router(auth.router)

@app.get("/")
def read_root():
  return {"Hello it is working?"}