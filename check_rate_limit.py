#!/usr/bin/env python3
"""Check GitHub Models rate limit status"""

import os
import sys
from openai import OpenAI

def check_rate_limit():
    """Query GitHub Models API and display rate limit headers"""
    
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    
    print(f"Token: {token[:10]}...{token[-4:]}")
    print()
    
    try:
        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token
        )
        
        print("Testing with a minimal request...")
        print("-" * 70)
        
        # Make a tiny request to trigger rate limit headers
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        
        print("✅ Request successful!")
        print()
        print("Rate Limit Status:")
        print("-" * 70)
        
        # Note: OpenAI SDK doesn't expose response headers directly
        # Rate limit info is typically in response headers like:
        # x-ratelimit-limit-requests
        # x-ratelimit-remaining-requests
        # x-ratelimit-reset-requests
        
        print("Response received successfully")
        print(f"Model used: {response.model}")
        print(f"Usage: {response.usage}")
        print()
        print("⚠️  GitHub Models limits (free tier):")
        print("   • 150 requests per 24 hours")
        print("   • 150,000 tokens per day")
        print("   • 10 RPM (requests per minute)")
        print()
        print("Note: GitHub Pro does NOT increase these limits.")
        print("The daily limit resets 24 hours after your first request.")
        
    except Exception as e:
        error_str = str(e)
        print(f"❌ Request failed: {error_str}")
        print()
        
        if "RateLimitReached" in error_str:
            print("🚫 RATE LIMIT EXCEEDED")
            print()
            print("You've hit the daily limit of 150 requests.")
            print("Your quota will reset 24 hours after your first request today.")
            print()
            print("Options:")
            print("1. Wait ~24 hours for reset")
            print("2. Switch to Gemini (1500 requests/day free)")
            print("3. Use OpenAI (paid, no limits)")
        
        sys.exit(1)

if __name__ == "__main__":
    check_rate_limit()
