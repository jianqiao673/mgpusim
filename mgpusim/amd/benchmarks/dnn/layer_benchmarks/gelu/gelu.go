// Package gelu implements the GELU algorithm as a benchmark for mgpusim v4.
package gelu

import (
	"log"
	// embed hsaco
	_ "embed"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/kernels"
)

// KernelArgs defines kernel arguments
type KernelArgs struct {
	Count               uint32
	Padding             uint32
	Input               driver.Ptr
	Output              driver.Ptr
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
}

// Benchmark defines a benchmark
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	hsaco   *insts.HsaCo

	Length      int
	inputData   []float32
	outputData  []float32
	gInputData  driver.Ptr
	gOutputData driver.Ptr

	useUnifiedMemory bool
	saveMemory       bool
}

//go:embed gelu.hsaco
var hsacoBytes []byte

// NewBenchmark returns a new Benchmark
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)
	b.driver = driver
	b.context = driver.Init()

	if len(hsacoBytes) == 0 {
		log.Panic("embedded gelu.hsaco is empty")
	}
	log.Printf("DEBUG: gelu.hsaco embedded size = %d bytes\n", len(hsacoBytes))

	b.hsaco = kernels.LoadProgramFromMemory(hsacoBytes, "GELUForward")
	b.Length = 1
	return b
}

// SelectGPU selects GPU(s)
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory enables unified memory
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// SetMemorySaving sets memory saving mode (ignored)
func (b *Benchmark) SetMemorySaving() {
	b.saveMemory = true
}

// Run runs the benchmark
func (b *Benchmark) Run() {
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

// initMem allocates memory and initializes input
func (b *Benchmark) initMem() {
	size := uint64(b.Length * 4)
	if b.useUnifiedMemory {
		b.gInputData = b.driver.AllocateUnifiedMemory(b.context, size)
		b.gOutputData = b.driver.AllocateUnifiedMemory(b.context, size)
	} else {
		b.gInputData = b.driver.AllocateMemory(b.context, size)
		b.driver.Distribute(b.context, b.gInputData, size, b.gpus)
		b.gOutputData = b.driver.AllocateMemory(b.context, size)
		b.driver.Distribute(b.context, b.gOutputData, size, b.gpus)
	}

	log.Printf("gInputData: 0x%x, gOutputData: 0x%x\n", b.gInputData, b.gOutputData)

	b.inputData = make([]float32, b.Length)
	b.outputData = make([]float32, b.Length)
	for i := 0; i < b.Length; i++ {
		b.inputData[i] = float32(i) - 0.5
	}

	b.driver.MemCopyH2D(b.context, b.gInputData, b.inputData)
}

// exec launches kernel and copies result back
func (b *Benchmark) exec() {
	queues := make([]*driver.CommandQueue, len(b.gpus))

	for i, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		q := b.driver.CreateCommandQueue(b.context)
		queues[i] = q

		numWI := b.Length / len(b.gpus)

		kernArg := KernelArgs{
			Count:               uint32(b.Length),
			Padding:             0,
			Input:               b.gInputData,
			Output:              b.gOutputData,
			HiddenGlobalOffsetX: int64(numWI * i),
		}

		b.driver.EnqueueLaunchKernel(
			q,
			b.hsaco,
			[3]uint32{uint32(numWI), 1, 1},
			[3]uint16{64, 1, 1},
			&kernArg,
		)
	}

	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}

	b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)

	// Free memory
	b.driver.FreeMemory(b.context, b.gInputData)
	b.driver.FreeMemory(b.context, b.gOutputData)
}

// Verify checks output against approximate GELU formula
func (b *Benchmark) Verify() {
	eps := float32(1e-4)
	c := float32(0.79788456) // sqrt(2/pi)
	for i := 0; i < b.Length; i++ {
		x := b.inputData[i]
		x3 := x * x * x
		y := 0.5 * x * (1 + c*(x + 0.044715*x3))
		diff := b.outputData[i] - y
		if diff < 0 {
			diff = -diff
		}
		if diff > eps {
			log.Panicf("mismatch at %d, input %f, output %f, expected %f",
				i, b.inputData[i], b.outputData[i], y)
		}
	}
	log.Printf("Passed!\n")
}


