#!/usr/bin/env python3
"""Get a Google OAuth refresh token with Drive + YouTube scopes.

Uses your existing secret.json which already has http://localhost registered.

Usage:
    python get_refresh_token.py
"""

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import json
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.upload",
]

print()
print("Opening browser for Google authorisation...")
print("Sign in with divysingh178@gmail.com and allow ALL permissions.")
print()

# Use secret.json directly — it already has http://localhost as redirect URI
flow = InstalledAppFlow.from_client_secrets_file("secret.json", scopes=SCOPES)

# port=0 lets the OS pick any free port on localhost — works with the
# http://localhost redirect URI registered in your OAuth client
creds = flow.run_local_server(
    port=0,
    prompt="consent",
    access_type="offline",
)

print()
print("=" * 70)
print("SUCCESS — copy this line into your .env file:")
print("=" * 70)
print()
print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
print()
print(f"Scopes granted: {sorted(creds.scopes)}")
