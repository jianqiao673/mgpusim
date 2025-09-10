#!/bin/bash
# run script for macOS

# 默认值
DEFAULT_PROGRAM="xor"
DEFAULT_ARGS="-timing -trace-vis"
# DEFAULT_ARGS="-timing -trace-vis -verify"
# DEFAULT_ARGS="-timing -trace-vis -save-memory"

# 帮助信息
usage() {
    echo "Usage: $0 [-p PROGRAM_NAME] [-a PROGRAM_ARGS]"
    echo "Options:"
    echo "  -p PROGRAM_NAME   Specify the program to run (default: $DEFAULT_PROGRAM)"
    echo "  -a PROGRAM_ARGS   Specify additional program arguments (default: '$DEFAULT_ARGS')"
    exit 1
}

# 解析参数
while getopts ":p:a:" opt; do
    case $opt in
        p) PROGRAM="$OPTARG" ;;
        a) ARGS="$OPTARG" ;;
        \?) usage ;;
    esac
done

# 使用默认值
PROGRAM=${PROGRAM:-$DEFAULT_PROGRAM}
ARGS=${ARGS:-$DEFAULT_ARGS}

# 检查目录
if [ ! -d "mgpusim/amd/samples/$PROGRAM" ]; then
    echo "Error: Directory mgpusim/amd/samples/$PROGRAM does not exist!" >&2
    exit 1
fi

# 创建日志目录
mkdir -p log

# 构建命令
CMD="cd mgpusim/amd/samples/$PROGRAM && rm -f akita_sim_*.sqlite3* && go build && ./$PROGRAM $ARGS"

# 打印命令
echo "Executing: $CMD" >&2
echo "Start time: $(date)" >&2

# 记录开始时间
START_TIME=$(date +%s.%N)

# 执行命令并重定向输出
eval "$CMD" &> "log/${PROGRAM}.txt"

# 记录结束时间
END_TIME=$(date +%s.%N)

# 计算耗时（跨平台）
ELAPSED_TIME=$(awk "BEGIN {print $END_TIME - $START_TIME}")

# 结果保存
DST="../../../../plot/data/$PROGRAM/"
mkdir -p "$DST"

# 移动结果文件（检查存在性）
FILES="akita_sim_*.sqlite3"
shopt -s nullglob 2>/dev/null || true   # Linux bash 支持, macOS 忽略
if compgen -G "$FILES" > /dev/null; then
    mv $FILES "$DST"
else
    echo "Warning: no akita_sim_*.sqlite3 files found, skipping move." >&2
fi

# 打印完成信息
echo "End time: $(date)" >&2
printf "Elapsed time: %.2f seconds\n" "$ELAPSED_TIME" >&2

cd ../../../../

# 日期格式化 (Linux vs macOS)
if date -d @"${START_TIME%.*}" >/dev/null 2>&1; then
    START_HUMAN=$(date -d @"${START_TIME%.*}")
    END_HUMAN=$(date -d @"${END_TIME%.*}")
else
    START_HUMAN=$(date -r "${START_TIME%.*}")
    END_HUMAN=$(date -r "${END_TIME%.*}")
fi

{
    echo "Start time: $START_HUMAN"
    echo "End time: $END_HUMAN"
    echo "Elapsed time: $ELAPSED_TIME seconds"
} >> "log/${PROGRAM}.txt"