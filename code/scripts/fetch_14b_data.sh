#!/bin/bash
set -e

# 从 westd-42896 拉取 exp_14b_c4 生成数据到 cqa1
WESTD_HOST="connect.westd.seetacloud.com"
WESTD_PORT=42896
WESTD_USER="root"
SRC_DIR="/root/autodl-tmp/gen-depth-contamination/data/exp_14b_c4"
DST="/root/autodl-tmp/gen-depth-contamination/data/exp_14b_c4/"

echo "=== Fetching exp_14b_c4 data from westd ==="

# 先检查 westd 上数据是否完整
echo "Checking remote data completeness..."
ssh -o StrictHostKeyChecking=no -p ${WESTD_PORT} ${WESTD_USER}@${WESTD_HOST} \
    "ls -la ${SRC_DIR}/depth_*.jsonl"

# 创建本地目录
mkdir -p ${DST}

# 拉取 depth 文件（不拉 seed_docs，太大且不需要）
echo "Transferring depth files..."
scp -o StrictHostKeyChecking=no -P ${WESTD_PORT} \
    ${WESTD_USER}@${WESTD_HOST}:${SRC_DIR}/depth_*.jsonl \
    ${DST}/

echo ""
echo "=== Transfer complete ==="
echo "Local files:"
ls -la ${DST}/depth_*.jsonl
echo ""
echo "File counts:"
for f in ${DST}/depth_*.jsonl; do
    echo "  $(basename $f): $(wc -l < $f) lines"
done
