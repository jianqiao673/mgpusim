package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layer_benchmarks/attention"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var (
	batchSize  = flag.Int("batch", 2, "Batch size")
	seqLen     = flag.Int("seqlen", 50, "Sequence length")
	embedDim   = flag.Int("embed", 8, "Embedding dimension")
	numHeads   = flag.Int("heads", 4, "Number of attention heads")
	saveMemory    = flag.Bool("lazy", false, "Use lazy memory allocation")
	useUnified = flag.Bool("unified", false, "Use unified memory")
)

func main() {
	flag.Parse()

	runner := new(runner.Runner).Init()
	benchmark := attention.NewBenchmark(runner.Driver())
	benchmark.SetParameters(*batchSize, *seqLen, *embedDim, *numHeads)

	if *saveMemory {
		benchmark.SetMemorySaving()
	}
	if *useUnified {
		benchmark.SetUnifiedMemory()
	}

	benchmark.SelectGPU([]int{0})

	runner.AddBenchmark(benchmark)

	runner.Run()
}
