"""Free Google Search grounding for ad-hoc lookups (install problems, library docs).

Kept separate from planner.py because grounding can't be combined with
JSON-schema output on many models.
"""
from google.genai import types

from .config import MODEL, client


def ground(question):
    r = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )
    return r.text