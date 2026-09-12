import os
import time
import json
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class ClassificationResult(BaseModel):
    category: str = Field(description="One of: billing, technical, account, general")
    summary: str = Field(description="One-sentence summary of the customer's issue")
    confidence: float = Field(default=0.95, description="Confidence score between 0.0 and 1.0")
    latency_ms: float = Field(default=0.0, description="Latency of response in milliseconds")
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

def load_prompt_config(prompt_path: str) -> Dict[str, Any]:
    """Load a versioned prompt YAML file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _mock_classify(email_text: str, prompt_version: str) -> ClassificationResult:
    """
    Deterministic mock classifier simulating LLM behavior based on prompt quality.
    - v1.0 / v1.2: Correct predictions for most test cases.
    - v1.1_regressed: Fails on edge cases (typos, sarcasm, mixed requests, account topics mapped to billing).
    """
    start_time = time.time()
    email_lower = email_text.lower()

    if "regressed" in prompt_version:
        # Regressed prompt behavior:
        # Fails account issues (maps to billing or technical), fails sarcasm, fails French
        if "bonjour" in email_lower or "parametres" in email_lower:
            cat = "general" # Failed translation
        elif "password" in email_lower or "unlock" in email_lower or "privileges" in email_lower:
            cat = "billing" # Regressed rule: account goes to billing
        elif "top notch" in email_lower or "great job" in email_lower:
            cat = "general" # Missed sarcasm crash
        elif "paid the annual plan" in email_lower:
            cat = "technical" # Misclassified mixed issue
        elif "card" in email_lower or "billed" in email_lower or "receipt" in email_lower or "cancel" in email_lower:
            cat = "billing"
        elif "freeze" in email_lower or "error" in email_lower or "screen" in email_lower:
            cat = "technical"
        else:
            cat = "general"
    else:
        # Quality prompt behavior (v1.0 & v1.2)
        if any(kw in email_lower for kw in ["card", "billed", "charge", "charged", "invoice", "tax", "paid", "cancel", "receipt"]):
            cat = "billing"
        elif "freeze" in email_lower or "crash" in email_lower or "error" in email_lower or "504" in email_lower or "403" in email_lower or "screen" in email_lower:
            cat = "technical"
        elif "password" in email_lower or "unlock" in email_lower or "email" in email_lower or "privileges" in email_lower or "team member" in email_lower or "bonjour" in email_lower:
            cat = "account"
        else:
            cat = "general"

    elapsed_ms = round((time.time() - start_time) * 1000 + 120, 2)
    summary = f"Issue regarding {cat}: {email_text[:60]}..."

    return ClassificationResult(
        category=cat,
        summary=summary,
        confidence=0.92 if "regressed" in prompt_version else 0.98,
        latency_ms=elapsed_ms,
        token_usage={"prompt_tokens": 115, "completion_tokens": 35, "total_tokens": 150}
    )

def classify_email(email_text: str, prompt_config: Dict[str, Any], use_mock: Optional[bool] = None) -> ClassificationResult:
    """Classify an email using OpenAI or Mock fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if use_mock is None:
        use_mock = not bool(api_key and api_key.startswith("sk-"))

    if use_mock:
        return _mock_classify(email_text, prompt_config.get("version", "v1.0"))

    # Live OpenAI API call
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system_prompt = prompt_config.get("system_prompt", "")
    temperature = prompt_config.get("temperature", 0.1)
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this email:\n\n{email_text}"}
            ],
            response_format={"type": "json_object"}
        )
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        content = json.loads(response.choices[0].message.content)
        usage = response.usage

        return ClassificationResult(
            category=content.get("category", "general").lower(),
            summary=content.get("summary", email_text[:80]),
            confidence=float(content.get("confidence", 0.9)),
            latency_ms=elapsed_ms,
            token_usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
        )
    except Exception as e:
        # Fallback to mock on error
        res = _mock_classify(email_text, prompt_config.get("version", "v1.0"))
        res.summary += f" (Fallback due to: {str(e)})"
        return res
