#!/bin/bash
# Smoke Test Suite for LangGraph Multi-Agent MCTS Framework Docker Deployment
# Usage: ./scripts/smoke_test.sh [PORT]

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:$PORT"
API_KEY="demo-api-key-replace-in-production"
PASS=0
FAIL=0

echo "=== MCTS Framework Docker Smoke Tests ==="
echo "Target: $BASE_URL"
echo "Started at: $(date)"
echo ""

test_endpoint() {
    local name="$1"
    local method="$2"
    local url="$3"
    local expected_code="$4"
    local data="$5"
    local headers="$6"

    echo -n "Test: $name... "

    if [ "$method" = "GET" ]; then
        code=$(curl -s -o /dev/null -w "%{http_code}" $headers "$url" 2>/dev/null)
    else
        code=$(curl -s -o /dev/null -w "%{http_code}" -X POST $headers -H "Content-Type: application/json" -d "$data" "$url" 2>/dev/null)
    fi

    if [ "$code" = "$expected_code" ]; then
        echo "PASS (HTTP $code)"
        ((PASS++))
    else
        echo "FAIL (Expected $expected_code, got $code)"
        ((FAIL++))
    fi
}

# Test 1: Health Check
test_endpoint "Health Check" "GET" "$BASE_URL/health" "200"

# Test 2: Readiness Check
test_endpoint "Readiness Check" "GET" "$BASE_URL/ready" "200"

# Test 3: OpenAPI Docs
test_endpoint "OpenAPI Docs" "GET" "$BASE_URL/docs" "200"

# Test 4: Query with Valid API Key
test_endpoint "Query (Valid Key)" "POST" "$BASE_URL/query" "200" \
    '{"query":"Test tactical scenario","use_mcts":false}' \
    "-H X-API-Key:$API_KEY"

# Test 5: Query with MCTS
test_endpoint "Query (with MCTS)" "POST" "$BASE_URL/query" "200" \
    '{"query":"Analyze defensive positions","use_mcts":true,"mcts_iterations":10}' \
    "-H X-API-Key:$API_KEY"

# Test 6: Authentication Failure
test_endpoint "Auth Failure" "POST" "$BASE_URL/query" "401" \
    '{"query":"Test"}' \
    "-H X-API-Key:invalid-key"

# Test 7: Validation Error (empty query)
test_endpoint "Validation Error" "POST" "$BASE_URL/query" "422" \
    '{"query":""}' \
    "-H X-API-Key:$API_KEY"

# Test 8: Metrics Endpoint (may return 501 if Prometheus not installed)
echo -n "Test: Metrics Endpoint... "
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/metrics" 2>/dev/null)
if [ "$code" = "200" ] || [ "$code" = "501" ]; then
    echo "PASS (HTTP $code - $([[ $code == '501' ]] && echo 'Not Configured' || echo 'Available'))"
    ((PASS++))
else
    echo "FAIL (Expected 200 or 501, got $code)"
    ((FAIL++))
fi

echo ""
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Total:  $((PASS + FAIL))"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "SUCCESS: All smoke tests passed!"
    exit 0
else
    echo "FAILURE: $FAIL test(s) failed"
    exit 1
fi
