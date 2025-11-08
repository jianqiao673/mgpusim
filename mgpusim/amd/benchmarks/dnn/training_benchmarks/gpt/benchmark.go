package gpt

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layers"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training/optimization"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

type Config struct {
	VocabSize int
	BlockSize int
	NEmbd     int
	NLayer    int
	NHeads    int
	Bias      bool
}

type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	to      *gputensor.GPUOperator

	gpus []int

	network training.Network
	trainer training.Trainer

	saveMemory bool

	config Config
}

// 创建 Benchmark，带日志
func NewBenchmark(driver *driver.Driver, saveMemory bool, config Config) *Benchmark {
	fmt.Println("[LOG] >>> Initializing Benchmark")

	b := new(Benchmark)
	b.driver = driver

	fmt.Println("[LOG] >>> Creating GPU context")
	b.context = b.driver.Init()

	fmt.Println("[LOG] >>> Creating GPU operator")
	b.to = gputensor.NewGPUOperator(b.driver, b.context)
	b.to.EnableVerification()

	b.config = config
	fmt.Printf("[LOG] >>> Config: %+v\n", b.config)

	if saveMemory {
		fmt.Println("[LOG] >>> Enabling memory saving mode")
		b.setMemorySaving()
	}

	fmt.Println("[LOG] >>> Building GPT network layers...")
	b.network = training.Network{
		Layers: []layers.Layer{
			layers.NewEmbeddingLayer("wte", 0, b.to, b.config.VocabSize, b.config.NEmbd),
			layers.NewEmbeddingLayer("wpe", 0, b.to, b.config.BlockSize, b.config.NEmbd),
			layers.NewTransformerLayerStack(b.to, b.config.NLayer, b.config.NEmbd, b.config.NHeads, false),
			layers.NewLayerNormLayer("ln_f", b.to, b.config.NEmbd),
			layers.NewBFullyConnectedLayer("lm_head", b.to, b.config.NEmbd, b.config.VocabSize, b.config.Bias),
		},
	}

	fmt.Println("[LOG] >>> GPT network built successfully with layers:", len(b.network.Layers))

	seq := []int{1, 1, 1, 1, 0, 1, 1, 1, 1, 0}
	contextLength := 3

	fmt.Println("[LOG] >>> Creating Trainer...")
	b.trainer = training.Trainer{
		TO:              b.to,
		DataSource:      NewDataSource(b.to, seq, contextLength),
		Network:         b.network,
		LossFunc:        training.NewSoftmaxCrossEntropy(b.to),
		OptimizationAlg: optimization.NewAdamW(b.to, 0.003, 0.1),
		Epoch:           1,
		BatchSize:       4,
		ShowBatchInfo:   true,
	}
	fmt.Println("[LOG] >>> Trainer created successfully")

	b.enableLayerVerification(&b.network)
	fmt.Println("[LOG] >>> Benchmark initialization finished")

	return b
}

func (b *Benchmark) enableLayerVerification(network *training.Network) {
	fmt.Println("[LOG] >>> Layer verification enabled (placeholder)")
}

func (b *Benchmark) SelectGPU(gpuIDs []int) {
	fmt.Printf("[LOG] >>> Selecting GPU(s): %v\n", gpuIDs)
	if len(gpuIDs) > 1 {
		panic("multi-GPU is not supported by DNN workloads")
	}
}

func (b *Benchmark) Run() {
	fmt.Println("[LOG] >>> Starting benchmark run")
	for i, l := range b.network.Layers {
		fmt.Printf("[LOG] >>> Initializing Layer %d: %T\n", i, l)
		if b.saveMemory {
			l.LazyRandomize()
		} else {
			l.Randomize()
		}
	}
	fmt.Println("[LOG] >>> All layers initialized successfully")

	if b.saveMemory {
		fmt.Println("[LOG] >>> Starting trainer.SaveTrain() ...")
		b.trainer.SaveTrain()
		fmt.Println("[LOG] >>> trainer.SaveTrain() finished")
	} else {
		fmt.Println("[LOG] >>> Starting trainer.Train() ...")
		b.trainer.Train()
		fmt.Println("[LOG] >>> trainer.Train() finished")
	}
}

func (b *Benchmark) printLayerParams() {
	fmt.Println("[LOG] >>> Printing layer parameters")
	for i, l := range b.network.Layers {
		params := l.Parameters()
		if params != nil {
			fmt.Println("Layer ", i, params.Vector())
		}
	}
}

func (b *Benchmark) Verify() {
	panic("not implemented")
}

func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}

func (b *Benchmark) setMemorySaving() {
	fmt.Println("[LOG] >>> Setting memory saving flag")
	b.saveMemory = true
}
