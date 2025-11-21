package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training_benchmarks/gpt"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

func main() {

	saveMemory := flag.Bool("save-mem", false , "Enable memory saving mode")
	flag.Parse()

	r := new(runner.Runner).Init()
	r.SaveMemory = *saveMemory
    
	benchmark := gpt.NewBenchmark(r.Driver(), r.SaveMemory, gpt.Config{
		BlockSize: 3,
		VocabSize: 2,
		NEmbd:     16,
		NLayer:    1,
		NHeads:    4,
		Bias:      false,
	})

	r.AddBenchmark(benchmark)

	r.Run()
}
