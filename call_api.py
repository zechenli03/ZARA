# !/usr/bin/env python
# -*-coding:utf-8 -*-

from google.genai.errors import ServerError
from google.genai import types
import time
import openai
import random


def call_gemini(client, sys_prompt, prompt, model_name, temp):
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=temp,
        )
    )
    return response.text


def call_gemini_with_retry(client, sys_prompt, prompt,
                           max_retry=10,
                           base_delay=1.0,   # Initial wait for 1 second
                           max_delay=30.0,
                           model_name="gemini-2.0-flash",
                           temp=0):  # Maximum waiting 30 seconds
    """
    Gemini call with retry and backoff.
    """
    attempt = 0
    while True:
        try:
            return call_gemini(client, sys_prompt, prompt, model_name, temp)
        except ServerError as e:
            if e.code == 503 or getattr(e, "status", "") == "UNAVAILABLE":
                attempt += 1
                if attempt > max_retry:
                    raise RuntimeError(f"Exceed the maximum number of retries ({max_retry})：{e}")
                delay = min(base_delay * 2 ** (attempt - 1), max_delay)
                print(f"[Retry {attempt}/{max_retry}] 503 model overloaded, sleeping {delay:.1f}s …")
                time.sleep(delay)
                continue
            # Throw out other errors
            raise


def call_gpt_with_retry(client, prompt,
                        max_retry=10, base_delay=1.0, max_delay=30.0,
                        model_name="gpt-4o-mini", temp=0):

    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp
            )
            return response.choices[0].message.content

        except (openai.APIError, openai.APIConnectionError,
                openai.RateLimitError) as e:
            print(f"[Attempt {attempt+1}] API Error: {e}")
            # Exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** attempt))
            jitter = random.uniform(0, delay)
            print(f"Retrying after {jitter:.2f} seconds...")
            time.sleep(jitter)

        except Exception as e:
            # Catch-all for unexpected errors
            print(f"[Attempt {attempt+1}] Unexpected error: {e}")
            return {"error": "Unexpected error: " + str(e)}

    return {"error": f"Failed after {max_retry} retries."}