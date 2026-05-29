# scripts/verify_setup.py
"""Run this after setup to confirm everything works."""
import os
from dotenv import load_dotenv
load_dotenv()

def verify():
    checks = []
    
    # Check API keys
    for key in ["GROQ_API_KEY", "COHERE_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key, "")
        status = "✅" if val and len(val) > 10 else "❌"
        checks.append(f"{status} {key}: {'set' if val else 'MISSING'}")
    
    # Check imports
    try:
        import langchain, chromadb, groq, cohere, rich
        checks.append("✅ Core packages installed")
    except ImportError as e:
        checks.append(f"❌ Missing package: {e}")
    
    # Test Groq connection
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say: SETUP OK"}],
            max_tokens=10
        )
        checks.append(f"✅ Groq API: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        checks.append(f"❌ Groq API: {e}")
    
    # Test Cohere
    try:
        import cohere
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        resp = co.embed(texts=["test"], model="embed-english-v3.0",
                       input_type="search_query", embedding_types=["float"])
        checks.append(f"✅ Cohere Embeddings: dim={len(resp.embeddings.float[0])}")
    except Exception as e:
        checks.append(f"❌ Cohere API: {e}")
    
    print("\n=== AlgoRAG Setup Verification ===")
    for check in checks:
        print(check)

if __name__ == "__main__":
    verify()