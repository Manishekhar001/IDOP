#!/usr/bin/env python3
"""
IDOP Post-Deployment Integration Smoke Test
-------------------------------------------
This script verifies the live deployment of the IDOP (Intelligent Data Operations Platform)
by executing real integration tests against the live API endpoints:
  1. Health check (/health)
  2. Vector collection info (/documents/info)
  3. Document ingestion (/documents/upload)
  4. Multi-agent query & RAG reasoning (/chat)
"""

import os
import sys
import uuid
import time
import requests

# Retrieve API base target URL from environment variable (injected by cd.yml)
API_URL = os.getenv("API_TARGET_URL", "http://localhost:8000")

# Global variables for testing flow
test_filename = f"smoke-test-{uuid.uuid4().hex[:8]}.txt"
test_doc_content = (
    f"IDOP Smoke Test Document\n"
    f"========================\n"
    f"Deployment Timestamp: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}\n"
    f"Security Verification Code: IDOP-SEC-9988X\n"
    f"Operational Status: Active and verified by automation.\n"
)
thread_id = str(uuid.uuid4())
user_id = "integration-smoke-test-user"


def run_health_check():
    print("🧪 Test 1: Verifying System Health...")
    endpoint = f"{API_URL}/health"
    try:
        response = requests.get(endpoint, timeout=30)
        print(f"   HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Health check failed with status: {response.status_code}")
            return False
            
        data = response.json()
        print(f"   Response: {data}")
        
        status = data.get("status")
        version = data.get("version")
        
        if status == "healthy":
            print(f"   ✅ Health endpoint verified! [Version: {version}]")
            print("")
            return True
        else:
            print(f"   ❌ Health endpoint reported degraded status: {status}")
            return False
            
    except Exception as e:
        print(f"   ❌ Network/Request error during health check: {e}")
        return False


def run_collection_info():
    print("🧪 Test 2: Verifying Vector Store Collection Info...")
    endpoint = f"{API_URL}/documents/info"
    try:
        response = requests.get(endpoint, timeout=30)
        print(f"   HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Collection info failed with status: {response.status_code}")
            return False
            
        data = response.json()
        print(f"   Response: {data}")
        
        collection_name = data.get("collection_name")
        total_documents = data.get("total_documents")
        status = data.get("status")
        
        print(f"   ✅ Vector store active: {collection_name} (Points: {total_documents}, Status: {status})")
        print("")
        return True
            
    except Exception as e:
        print(f"   ❌ Network/Request error during collection check: {e}")
        return False


def run_document_upload():
    print(f"🧪 Test 3: Uploading Temp Document ({test_filename})...")
    endpoint = f"{API_URL}/documents/upload"
    
    # Create temporary text file locally
    with open(test_filename, "w", encoding="utf-8") as f:
        f.write(test_doc_content)
        
    try:
        # Perform multipart file upload
        with open(test_filename, "rb") as f:
            files = {"file": (test_filename, f, "text/plain")}
            response = requests.post(endpoint, files=files, timeout=45)
            
        print(f"   HTTP Status: {response.status_code}")
        
        # Clean up temp file
        if os.path.exists(test_filename):
            os.remove(test_filename)
            
        if response.status_code not in (200, 201):
            print(f"   ❌ Document upload failed with status: {response.status_code}")
            print(f"   Response Body: {response.text}")
            return False
            
        data = response.json()
        print(f"   Response Chunks Created: {data.get('chunks_created')}")
        print(f"   Indexed Point IDs Count: {len(data.get('document_ids', []))}")
        
        if data.get("chunks_created", 0) > 0:
            print("   ✅ Document successfully parsed, embedded, and indexed in Qdrant!")
            print("")
            return True
        else:
            print("   ❌ Document uploaded but no vector chunks were created!")
            return False
            
    except Exception as e:
        if os.path.exists(test_filename):
            os.remove(test_filename)
        print(f"   ❌ Network/Request error during document upload: {e}")
        return False


def run_chat_query():
    print("🧪 Test 4: Querying LangGraph RAG Multi-Agent Pipeline...")
    endpoint = f"{API_URL}/chat"
    
    # Construct ChatRequest schema body
    payload = {
        "question": "What is the security verification code mentioned in the smoke test document?",
        "thread_id": thread_id,
        "user_id": user_id,
        "include_sources": True,
        "search_mode": "hybrid",
        "top_k": 3,
        "enable_hyde": False,
        "enable_reranking": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(endpoint, json=payload, timeout=60)
        latency = (time.time() - start_time) * 1000
        
        print(f"   HTTP Status: {response.status_code}")
        print(f"   Network roundtrip time: {latency:.2f} ms")
        
        if response.status_code != 200:
            print(f"   ❌ Chat query failed with status: {response.status_code}")
            print(f"   Response Body: {response.text}")
            return False
            
        data = response.json()
        answer = data.get("answer", "")
        crag_verdict = data.get("crag_verdict", "")
        issup = data.get("issup", "")
        sources = data.get("sources", [])
        
        print("\n   [RAG PIPELINE EXECUTION SUMMARY]")
        print(f"   ├─ CRAG Verdict      : {crag_verdict}")
        print(f"   ├─ SRAG Support      : {issup}")
        print(f"   └─ Source Chunks     : {len(sources)} items retrieved")
        print("\n   [AGENT RESPONSE]")
        print(f"   {answer}\n")
        
        # Verify the answer contains the correct validation keyword
        verification_code = "IDOP-SEC-9988X"
        if verification_code in answer:
            print("   ✅ Multi-Agent RAG reasoning completed and verified answer successfully!")
            print("")
            return True
        else:
            print(f"   ❌ RAG failed to retrieve or state verification code: {verification_code}")
            print("   (Verify the dense/sparse hybrid embeddings indexing process)")
            return False
            
    except Exception as e:
        print(f"   ❌ Network/Request error during chat query: {e}")
        return False


def run_tests():
    global API_URL
    if not os.getenv("API_TARGET_URL"):
        print("❌ ERROR: API_TARGET_URL environment variable is not set!")
        sys.exit(1)
        
    API_URL = os.getenv("API_TARGET_URL")
    
    print("==================================================")
    print("🚀 STARTING IDOP POST-DEPLOYMENT SMOKE TEST SUITE")
    print("==================================================")
    print(f"📡 API Endpoint Target : {API_URL}")
    print(f"⏱️  Timestamp          : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("==================================================\n")

    # Execute the tests sequentially
    tests = [
        run_health_check,
        run_collection_info,
        run_document_upload,
        run_chat_query
    ]
    
    success = True
    for test in tests:
        if not test():
            success = False
            break
            
    print("==================================================")
    if success:
        print("🎉 SUCCESS: IDOP Integration Smoke Tests PASSED!")
        print("   Production environment is active and performing perfectly.")
        print("==================================================")
        sys.exit(0)
    else:
        print("❌ FAILURE: IDOP Integration Smoke Tests FAILED!")
        print("   Please check the deployment logs and server instances.")
        print("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    # Give a small warm-up time before launching requests to let services fully start up
    print("⏳ Warming up connection (3 seconds)...")
    time.sleep(3)
    run_tests()
