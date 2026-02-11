#!/bin/bash
# Demo script for DevKnowledge

echo "🚀 DevKnowledge Demo"
echo "===================="
echo ""

# Initialize knowledge base in temp location
export DB_PATH="/tmp/devknowledge_demo.db"
rm -f $DB_PATH

echo "1. Initializing knowledge base..."
dk init --db-path $DB_PATH

echo ""
echo "2. Adding some notes..."

dk add note "Python AsyncIO" \
  --content "AsyncIO is Python's built-in library for writing concurrent code using async/await syntax" \
  --tags "python,async,concurrency" \
  --db-path $DB_PATH

dk add note "JavaScript Promises" \
  --content "Promises represent the eventual completion or failure of an asynchronous operation" \
  --tags "javascript,async,promises" \
  --db-path $DB_PATH

dk add note "REST API Best Practices" \
  --content "Use HTTP verbs correctly: GET for retrieval, POST for creation, PUT for updates, DELETE for removal" \
  --tags "api,rest,http" \
  --db-path $DB_PATH

echo ""
echo "3. Adding a code snippet..."

cat > /tmp/quicksort.py << 'EOF'
def quicksort(arr):
    """Quick sort implementation in Python."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
EOF

dk add code python "Quicksort Algorithm" \
  --file /tmp/quicksort.py \
  --tags "algorithm,sorting,python" \
  --db-path $DB_PATH

echo ""
echo "4. Searching for 'async'"
dk search "async" --db-path $DB_PATH

echo ""
echo "5. Viewing stats..."
dk stats --db-path $DB_PATH

echo ""
echo "6. Listing all documents..."
dk list --db-path $DB_PATH

echo ""
echo "7. Finding related documents to 'Python AsyncIO' (ID: 1)"
dk related 1 --db-path $DB_PATH

echo ""
echo "✅ Demo complete!"
echo "Database created at: $DB_PATH"
echo "Try: dk show 1 --db-path $DB_PATH"
