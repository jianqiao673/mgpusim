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
	NEmbd int
	NLayer int
	NHeads int
	Bias bool
}

// Benchmark defines the XOR network training benchmark.
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

// NewBenchmark creates a new benchmark.
func NewBenchmark(driver *driver.Driver, saveMemory bool, config Config) *Benchmark {
	b := new(Benchmark)

	b.driver = driver
	b.context = b.driver.Init()
	b.to = gputensor.NewGPUOperator(b.driver, b.context)
	b.to.EnableVerification()
	b.config = config

	if saveMemory {
		b.setMemorySaving()
	}

	if b.saveMemory {
		// TODO
	} else {
		b.network = training.Network{
			Layers: []layers.Layer{
				// Token embedding
				layers.NewEmbeddingLayer(
					"wte",
					b.to,
					b.config.VocabSize,
					b.config.NEmbd,
				),

				// Positional embedding
				layers.NewEmbeddingLayer(
					"wpe",
					b.to,
					b.config.BlockSize,
					b.config.NEmbd,
				),

				// Transformer blocks (h)
				NewTransformerLayerStack(
					b.to,
					b.config.NLayer,
					b.config.NEmbd,
					b.config.NumHeads,
				),

				// Final LayerNorm
				layers.NewLayerNormLayer(
					"ln_f",
					b.to,
					b.config.NEmbd,
				),

				// LM Head projection (weight tied to wte)
				layers.NewFullyConnectedLayer(
					"lm_head",
					b.to,
					b.config.NEmbd,
					b.config.VocabSize,
					b.config.Bias,
				),
			},
		}
	}

	b.trainer = training.Trainer{
		TO:              b.to,
		DataSource:      NewDataSource(b.to),
		Network:         b.network,
		LossFunc:        training.NewSoftmaxCrossEntropy(b.to),
		OptimizationAlg: optimization.NewAdamW(b.to, 0.003, 0.1),
		Epoch:           1, // default: 50
		// BatchSize:       20, // default: 4
		ShowBatchInfo:   true,
	}

	b.enableLayerVerification(&b.network)

	return b
}

func (b *Benchmark) enableLayerVerification(network *training.Network) {

}

// SelectGPU selects the GPU to use.
func (b *Benchmark) SelectGPU(gpuIDs []int) {
	if len(gpuIDs) > 1 {
		panic("multi-GPU is not supported by DNN workloads")
	}
}

// Run executes the benchmark.
func (b *Benchmark) Run() {
	for _, l := range b.network.Layers {
		if b.saveMemory {
			l.LazyRandomize()
		} else {
			l.Randomize()
		}
	}
	if b.saveMemory {
		b.trainer.SaveTrain()
	} else {
		b.trainer.Train()
	}
}

func (b *Benchmark) printLayerParams() {
	for i, l := range b.network.Layers {
		params := l.Parameters()
		if params != nil {
			fmt.Println("Layer ", i, params.Vector())
		}
	}
}

// Verify runs the benchmark on the CPU and checks the result.
func (b *Benchmark) Verify() {
	panic("not implemented")
}

// SetUnifiedMemory asks the benchmark to use unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}

// SetMemorySaving sets the memory saving mode
func (b *Benchmark) setMemorySaving() {
	b.saveMemory = true
}