package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training_benchmarks/gpt"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

func main() {
	flag.Parse()

	runner := new(runner.Runner).Init()

	benchmark := gpt.NewBenchmark(runner.Driver(), runner.SaveMemory, gpt.Config{
		BlockSize: 3,
		VocabSize: 2,
		NEmbd:     16,
		NLayer:    1,
		NHeads:    4,
		Bias:      false,
	})

	runner.AddBenchmark(benchmark)

	runner.Run()
}
