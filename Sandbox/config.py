"""Shared settings and the Gemini client."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GH_TOKEN = os.getenv("GITHUB_TOKEN")
KAGGLE_USER = os.getenv("KAGGLE_USERNAME", "saurabingh")
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=os.getenv("API_KEY"))