"""Shared settings and the Gemini client."""
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **kw):
        return False

try:
    from google import genai
except ImportError:
    genai = None

load_dotenv()

GH_TOKEN = os.getenv("GITHUB_TOKEN")
KAGGLE_USER = os.getenv("KAGGLE_USERNAME", "saurabingh")
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=os.getenv("API_KEY")) if (genai is not None and os.getenv("API_KEY")) else None