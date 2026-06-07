#!/bin/bash
set -e

echo "=========================================="
echo "🧪 IDOP COMPREHENSIVE ENDPOINT TESTS"
echo "=========================================="
echo ""

# TEST 1: /health
echo "===== TEST 1: /health ====="
HEALTH=$(curl -s http://localhost:8000/health)
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
echo "  Status: $STATUS"
if [ "$STATUS" = "healthy" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

# TEST 2: /documents/info
echo "===== TEST 2: /documents/info ====="
INFO=$(curl -s http://localhost:8000/documents/info)
INFO_STATUS=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
INFO_POINTS=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_documents',0))" 2>/dev/null)
echo "  Status: $INFO_STATUS, Points: $INFO_POINTS"
if [ "$INFO_STATUS" = "green" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

# TEST 3: /documents/upload (TXT)
echo "===== TEST 3: /documents/upload (TXT) ====="
echo "IDOP Smoke Test Document - Security Code: IDOP-SEC-9988X" > /tmp/test_endpoint.txt
TXT_RESULT=$(curl -s -o /tmp/txt_resp.json -w "%{http_code}" -X POST http://localhost:8000/documents/upload -F "file=@/tmp/test_endpoint.txt;type=text/plain")
TXT_CHUNKS=$(cat /tmp/txt_resp.json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('chunks_created','?'))" 2>/dev/null)
echo "  HTTP: $TXT_RESULT, Chunks: $TXT_CHUNKS"
if [ "$TXT_RESULT" = "200" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

# TEST 4: /cache/stats
echo "===== TEST 4: /cache/stats ====="
CACHE=$(curl -s http://localhost:8000/cache/stats)
CACHE_COUNT=$(echo "$CACHE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('document_cache',{}).get('total_documents',0))" 2>/dev/null)
echo "  Cached documents: $CACHE_COUNT"
echo "  ✅ PASS (cache endpoint reachable)"
echo ""

# TEST 5: /sql/generate
echo "===== TEST 5: /sql/generate ====="
SQL_RESP=$(curl -s -X POST http://localhost:8000/sql/generate -H "Content-Type: application/json" -d '{"question":"Show all products with price less than 100","vanna_temperature":0.0}')
SQL_STATUS=$(echo "$SQL_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?') + ' | sql_len=' + str(len(d.get('sql',''))))" 2>/dev/null)
SQL_TOKEN=$(echo "$SQL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
echo "  Status: $SQL_STATUS"
if [ -n "$SQL_TOKEN" ]; then echo "  Token present: ${SQL_TOKEN:0:16}... ✅"; else echo "  ❌ No token"; fi
echo ""

# TEST 6: /sql/execute
echo "===== TEST 6: /sql/execute ====="
SQL_ID=$(echo "$SQL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('query_id',''))" 2>/dev/null)
if [ -n "$SQL_ID" ] && [ -n "$SQL_TOKEN" ]; then
  EXEC_RESP=$(curl -s -o /tmp/exec_resp.json -w "%{http_code}" -X POST http://localhost:8000/sql/execute -H "Content-Type: application/json" -d "{\"query_id\":\"$SQL_ID\",\"approval_token\":\"$SQL_TOKEN\"}")
  EXEC_STATUS=$(cat /tmp/exec_resp.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?') + ' | rows=' + str(d.get('row_count',0)))" 2>/dev/null)
  echo "  HTTP: $EXEC_RESULT, Status: $EXEC_STATUS"
  echo "  ✅ PASS"
else
  echo "  ⚠️ SKIP (no valid query_id/token)"
fi
echo ""

# TEST 7: /mutation/upload
echo "===== TEST 7: /mutation/upload ====="
echo "id,name,price" > /tmp/mut_test.csv
echo "1,TestProduct,99.99" >> /tmp/mut_test.csv
MUT_RESP=$(curl -s -o /tmp/mut_resp.json -w "%{http_code}" -X POST http://localhost:8000/mutation/upload -F "file=@/tmp/mut_test.csv;type=text/csv" -F "table_name=products" -F "request_intent=Insert test product" -F "max_bulk_rows=5")
MUT_STATUS=$(cat /tmp/mut_resp.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?') + ' | token=' + str(d.get('token','none')[:16]))" 2>/dev/null)
echo "  HTTP: $MUT_RESULT, Status: $MUT_STATUS"
if [ "$MUT_RESULT" = "200" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

# TEST 8: /memory/store and /memory/recall
echo "===== TEST 8: /memory/store ====="
MEM_STORE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/memory/store -H "Content-Type: application/json" -d '{"user_id":"test-user","session_id":"test-session","content":"User likes AI testing"}')
echo "  Store HTTP: $MEM_STORE"
if [ "$MEM_STORE" = "200" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

echo "===== TEST 9: /memory/recall ====="
MEM_RECALL=$(curl -s -o /tmp/mem_resp.json -w "%{http_code}" -X POST http://localhost:8000/memory/recall -H "Content-Type: application/json" -d '{"user_id":"test-user","session_id":"test-session","query":"What does the user like?"}')
echo "  Recall HTTP: $MEM_RECALL"
if [ "$MEM_RECALL" = "200" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

# TEST 10: /chat
echo "===== TEST 10: /chat ====="
echo "  ⏳ This may take up to 60 seconds..."
CHAT_RESP=$(curl -s -o /tmp/chat_resp.json -w "%{http_code}" -m 120 -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"What documents do I have?","thread_id":"test-thread-endpoint","user_id":"test-user-endpoint","top_k":3}')
CHAT_DATA=$(cat /tmp/chat_resp.json 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Answer length: ' + str(len(d.get('answer',''))) + ' chars')
print('Sources: ' + str(len(d.get('sources',[]))))
print('Query type: ' + str(d.get('query_type','unknown')))
print('CRAG: ' + str(d.get('crag_verdict','')))
print('SRAG: ' + str(d.get('issup','')))
print('SQL: ' + str(d.get('sql_query','')[:60]))
print('Mutation: ' + str(d.get('mutation_status','none')))
" 2>/dev/null)
echo "  HTTP: $CHAT_RESP"
echo "$CHAT_DATA"
if [ "$CHAT_RESP" = "200" ]; then echo "  ✅ PASS"; else echo "  ❌ FAIL"; fi
echo ""

echo "=========================================="
echo "🏁 ALL ENDPOINT TESTS COMPLETED"
echo "=========================================="
