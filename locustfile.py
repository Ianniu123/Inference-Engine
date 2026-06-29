# locust -f locustfile.py --headless -u 16 -r 8 -t 20s --host http://127.0.0.1:8000

import random

from locust import HttpUser, between, task

MAX_TOKENS = 24
PROMPTS = [
    "The capital of France is", "Once upon a time there was", "In a distant galaxy",
    "The stock market today", "Scientists have discovered that", "My favorite food is",
    "The weather forecast says", "Deep learning models can", "The history of Rome",
    "A recipe for disaster",
]


class GenerateUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={"prompt": random.choice(PROMPTS), "max_tokens": MAX_TOKENS, "temperature": 0.0},
        )
