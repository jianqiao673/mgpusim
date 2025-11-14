package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layer_benchmarks/attention"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

// 定义命令行参数
var (
	batchSize  = flag.Int("batch", 2, "Batch size")
	seqLen     = flag.Int("seqlen", 50, "Sequence length")
	embedDim   = flag.Int("embed", 8, "Embedding dimension")
	numHeads   = flag.Int("heads", 4, "Number of attention heads")
	useLazy    = flag.Bool("lazy", true, "Use lazy memory allocation")
	useUnified = flag.Bool("unified", false, "Use unified memory")
)

func main() {
	flag.Parse()

	runner := new(runner.Runner).Init()

	// 创建Attention benchmark
	benchmark := attention.NewBenchmark(runner.Driver())

	// 设置Attention参数
	benchmark.SetParameters(*batchSize, *seqLen, *embedDim, *numHeads)

	// 设置内存模式
	if *useLazy {
		benchmark.SetMemorySaving()
	}
	if *useUnified {
		benchmark.SetUnifiedMemory()
	}

	// 选择GPU (默认使用GPU 0)
	benchmark.SelectGPU([]int{0})

	runner.AddBenchmark(benchmark)

	runner.Run()
}
