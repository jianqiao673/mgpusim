package main

import (
	"flag"
	"log"
	"strconv"
	"strings"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layer_benchmarks/embedding"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var (
	vocabSizeFlag = flag.Int("vocab", 50, "Vocabulary size (use small values for quick tests)")
	embedDimFlag  = flag.Int("dim", 8, "Embedding dimension (use small values for quick tests)")
	batchSizeFlag = flag.Int("batch", 2, "Batch size")
	seqLenFlag    = flag.Int("seq", 4, "Sequence length")

	// extras
	useUnifiedFlag = flag.Bool("unified", false, "Use unified memory for allocations")
	saveMemFlag    = flag.Bool("savemem", false, "Enable memory-saving (lazy) mode")
	gpuListFlag = flag.String("gpu-list", "0", "Comma-separated GPU ids to use, e.g. \"0\" or \"0,1\"")
)

func parseGPUs(s string) []int {
	s = strings.TrimSpace(s)
	if s == "" {
		return []int{0}
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			log.Printf("warning: invalid gpu id '%s', skipping", p)
			continue
		}
		out = append(out, id)
	}
	if len(out) == 0 {
		return []int{0}
	}
	return out
}

func main() {
	flag.Parse()

	r := new(runner.Runner).Init()

	bench := embedding.NewBenchmark(r.Driver())

	bench.SetParameters(*vocabSizeFlag, *embedDimFlag, *batchSizeFlag, *seqLenFlag)

	if *useUnifiedFlag {
		bench.SetUnifiedMemory()
	}
	if *saveMemFlag {
		bench.SetMemorySaving()
	}

	// 解析并设置 GPU 列表
	gpus := parseGPUs(*gpuListFlag)
	bench.SelectGPU(gpus)

	r.AddBenchmark(bench)
	r.Run()
}
