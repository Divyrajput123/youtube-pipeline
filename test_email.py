#!/usr/bin/env python3
"""Quick test to verify SMTP email is working."""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

host     = "smtp.gmail.com"
port     = 587
username = "divysingh178@gmail.com"
password = "gwmg ysyg xnbn ysqo"
to       = "divysingh178@gmail.com"

print(f"Connecting to {host}:{port}...")

msg = MIMEMultipart("alternative")
msg["Subject"] = "AI Pipeline — Test Email ✅"
msg["From"]    = username
msg["To"]      = to

plain = "This is a test from your AI YouTube Pipeline. Email is working!"

html = """<!DOCTYPE html>
<html><body style="font-family:-apple-system,sans-serif;padding:24px;background:#f9fafb">
<div style="max-width:560px;margin:0 auto;background:#fff;border-radius:16px;padding:32px;
            box-shadow:0 2px 16px rgba(0,0,0,.07)">
  <h2 style="color:#111827">✅ Email is working!</h2>
  <p style="color:#374151">Your AI YouTube Pipeline can send notifications.</p>
  <p style="color:#374151">When a script is ready for review, you will get an
  email like this with two tap buttons:</p>
  <div style="margin:28px 0;text-align:center">
    <a href="#approve" style="background:#16a34a;color:#fff;padding:16px 32px;
       border-radius:10px;text-decoration:none;font-size:16px;font-weight:bold;
       margin-right:12px">✅ Approve</a>
    <a href="#edit" style="background:#d97706;color:#fff;padding:16px 32px;
       border-radius:10px;text-decoration:none;font-size:16px;font-weight:bold">
       ✏️ Request Edits</a>
  </div>
  <p style="color:#9ca3af;font-size:13px">AI YouTube Content Pipeline</p>
</div>
</body></html>"""

msg.attach(MIMEText(plain, "plain"))
msg.attach(MIMEText(html, "html"))

try:
    with smtplib.SMTP(host, port) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(username, password)
        s.sendmail(username, [to], msg.as_string())
    print(f"✓ Email sent to {to} — check your inbox (and spam folder)!")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
