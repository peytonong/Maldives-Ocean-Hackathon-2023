from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from typing import List, Optional
from os import environ as env
# from schemas import Post
# from . import auth

router = APIRouter(
  prefix="/test",
  tags=['Test']
)

@router.get("/")
def get_posts():
    return {"Test Post!"}