#!/bin/bash
export INTEGRATED_AGENT_SNOMED_PATH="/Users/vic/Desktop/NICE/CAM_C04_NICE_Project/snomed_master_v3.csv"
export INTEGRATED_AGENT_CHROMA_DIR="/tmp/test_chroma_db"
export INTEGRATED_AGENT_EMBEDDINGS_DIR="/tmp/test_embeddings"
export INTEGRATED_AGENT_LLM_MODEL="llama3.1"

echo "=== Test 4: Both missing ==="
rm -rf /tmp/test_chroma_db /tmp/test_embeddings
python main.py | grep -E "Chroma DB|Embeddings resource|Embeddings directory|Loading existing"

echo "=== Test 1: Both exist ==="
python main.py | grep -E "Chroma DB|Embeddings resource|Embeddings directory|Loading existing"

echo "=== Test 2: Chroma missing, embeddings exist ==="
rm -rf /tmp/test_chroma_db
python main.py | grep -E "Chroma DB|Embeddings resource|Embeddings directory|Loading existing"

echo "=== Test 3: Chroma exists, embeddings missing ==="
# It was just recreated in Test 2, so it exists. Now delete embeddings:
rm -rf /tmp/test_embeddings
python main.py | grep -E "Chroma DB|Embeddings resource|Embeddings directory|Loading existing"
