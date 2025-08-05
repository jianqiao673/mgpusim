#!/bin/bash

# 设置默认值
DEFAULT_PROGRAM="xor"
# DEFAULT_ARGS="-timing -trace-vis"
# DEFAULT_ARGS="-timing -trace-vis -verify"
DEFAULT_ARGS="-timing -trace-vis -save-memory"

# 显示帮助信息
usage() {
    echo "Usage: $0 [-p PROGRAM_NAME] [-a PROGRAM_ARGS]"
    echo "Options:"
    echo "  -p PROGRAM_NAME   Specify the program to run (default: $DEFAULT_PROGRAM)"
    echo "  -a PROGRAM_ARGS   Specify additional program arguments (default: '$DEFAULT_ARGS')"
    exit 1
}

# 解析命令行参数
while getopts ":p:a:" opt; do
    case $opt in
        p) PROGRAM="$OPTARG" ;;
        a) ARGS="$OPTARG" ;;
        \?) usage ;;
    esac
done

# 如果没有提供参数，使用默认值
PROGRAM=${PROGRAM:-$DEFAULT_PROGRAM}
ARGS=${ARGS:-$DEFAULT_ARGS}

# 检查目录是否存在
if [ ! -d "mgpusim/amd/samples/$PROGRAM" ]; then
    echo "Error: Directory mgpusim/amd/samples/$PROGRAM does not exist!" >&2
    exit 1
fi

# 创建日志目录（如果不存在）
mkdir -p log

# 构建完整命令
CMD="cd mgpusim/amd/samples/$PROGRAM && rm -f akita_sim_**.sqlite3 && go build && ./$PROGRAM $ARGS"

# 打印命令（会显示在终端，不会被重定向）
echo "Executing: $CMD" >&2
echo "Start time: $(date)" >&2

# 记录开始时间
START_TIME=$(date +%s.%N)

# 执行命令并重定向输出
eval "$CMD" &> "log/${PROGRAM}.txt"

# 记录结束时间
END_TIME=$(date +%s.%N)

# 计算运行时间
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# 将运行结果移动到plot/data
DST="../../../../plot/data/$PROGRAM/"
mkdir -p $DST
mv akita_sim_**.sqlite3 $DST

# 打印完成信息
# echo "Execution completed." >&2
# echo "Output saved to log/${PROGRAM}.txt" >&2
echo "End time: $(date)" >&2
printf "Elapsed time: %.2f seconds\n" $ELAPSED_TIME >&2

cd ../../../../
# 将时间信息也写入日志文件
echo "Start time: $(date -d @${START_TIME%.*})" >> "log/${PROGRAM}.txt"
echo "End time: $(date -d @${END_TIME%.*})" >> "log/${PROGRAM}.txt"
echo "Elapsed time: $ELAPSED_TIME seconds" >> "log/${PROGRAM}.txt"
