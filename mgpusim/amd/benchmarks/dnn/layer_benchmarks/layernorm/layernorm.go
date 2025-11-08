// Package layernorm implements a small LayerNorm benchmark for mgpusim v4
package layernorm

import (
	_ "embed"
	"log"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/kernels"
)

// KernelArgs matches the HSACO kernel signature:
// input (ptr), output (ptr), count (uint32), hidden_size (uint32)
type KernelArgs struct {
	Input      uint64
	Output     uint64
	Count      uint32
	HiddenSize uint32
}

// Benchmark defines a benchmark
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	hsaco   *insts.HsaCo

	Length      int
	HiddenSize  int
	inputData   []float32
	outputData  []float32
	gInputData  driver.Ptr
	gOutputData driver.Ptr

	useUnifiedMemory bool
	saveMemory       bool
}

//go:embed layernorm.hsaco
var hsacoBytes []byte

// NewBenchmark creates a new benchmark with length and hiddenSize
func NewBenchmark(driver *driver.Driver, length int, hiddenSize int) *Benchmark {
	b := new(Benchmark)
	b.driver = driver
	b.context = driver.Init()

	if len(hsacoBytes) == 0 {
		log.Panic("embedded layernorm.hsaco is empty")
	}
	log.Printf("DEBUG: layernorm.hsaco embedded size = %d bytes\n", len(hsacoBytes))

	b.hsaco = kernels.LoadProgramFromMemory(hsacoBytes, "LayerNormForward")
	b.Length = length
	b.HiddenSize = hiddenSize
	return b
}

// SelectGPU selects GPUs to run
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory enables unified memory
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// SetMemorySaving enables memory saving mode
func (b *Benchmark) SetMemorySaving() {
	b.saveMemory = true
}

// Run executes the benchmark
func (b *Benchmark) Run() {
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

// initMem allocates memory and initializes input
func (b *Benchmark) initMem() {
	size := uint64(b.Length * b.HiddenSize * 4) // float32
	if b.useUnifiedMemory {
		b.gInputData = b.driver.AllocateUnifiedMemory(b.context, size)
		b.gOutputData = b.driver.AllocateUnifiedMemory(b.context, size)
	} else {
		b.gInputData = b.driver.AllocateMemory(b.context, size)
		b.driver.Distribute(b.context, b.gInputData, size, b.gpus)
		b.gOutputData = b.driver.AllocateMemory(b.context, size)
		b.driver.Distribute(b.context, b.gOutputData, size, b.gpus)
	}

	b.inputData = make([]float32, b.Length*b.HiddenSize)
	b.outputData = make([]float32, b.Length*b.HiddenSize)
	for i := range b.inputData {
		b.inputData[i] = float32(i)/float32(b.HiddenSize) - 0.5
	}

	b.driver.MemCopyH2D(b.context, b.gInputData, b.inputData)
	log.Printf("gInputData: 0x%x, gOutputData: 0x%x\n", b.gInputData, b.gOutputData)
}

// exec launches kernel and copies results back
func (b *Benchmark) exec() {
	queues := make([]*driver.CommandQueue, len(b.gpus))

	numGPUs := len(b.gpus)
	if numGPUs == 0 {
		log.Panic("no GPUs selected")
	}

	baseRows := b.Length / numGPUs
	remainder := b.Length % numGPUs

	for i, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		q := b.driver.CreateCommandQueue(b.context)
		queues[i] = q

		// rows assigned to this GPU
		rowsForThisGPU := baseRows
		if i == numGPUs-1 {
			rowsForThisGPU += remainder
		}

		// elements (total work-items) = rows * hidden_size
		elems := uint32(rowsForThisGPU * b.HiddenSize)
		if elems == 0 {
			log.Printf("GPU %d assigned 0 elements, skipping", gpu)
			continue
		}

		// byte size per row (float32)
		rowBytes := uint64(b.HiddenSize) * 4

		// compute offsetRows (sum rows of previous GPUs)
		offsetRows := uint64(0)
		for j := 0; j < i; j++ {
			offset := uint64(baseRows)
			if j == numGPUs-1 {
				offset += uint64(remainder)
			}
			offsetRows += offset
		}

		inputPtr := uint64(b.gInputData) + offsetRows*rowBytes
		outputPtr := uint64(b.gOutputData) + offsetRows*rowBytes

		kernArg := KernelArgs{
			Input:      inputPtr,
			Output:     outputPtr,
			Count:      uint32(rowsForThisGPU),
			HiddenSize: uint32(b.HiddenSize),
		}

		log.Printf("Enqueue GPU %d: rows=%d elems=%d input=0x%x output=0x%x",
			gpu, rowsForThisGPU, elems, kernArg.Input, kernArg.Output)

		b.driver.EnqueueLaunchKernel(
			q,
			b.hsaco,
			[3]uint32{elems, 1, 1},
			[3]uint16{64, 1, 1},
			&kernArg,
		)
	}

	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}

	b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)
	b.driver.FreeMemory(b.context, b.gInputData)
	b.driver.FreeMemory(b.context, b.gOutputData)
}

// Verify checks correctness (approximate, simple)
func (b *Benchmark) Verify() {
	eps := float32(1e-4)
	for i := 0; i < b.Length*b.HiddenSize; i++ {
		x := b.inputData[i]
		y := b.outputData[i]
		if y < x-eps || y > x+eps {
			log.Printf("i=%d, input=%f, output=%f\n", i, x, y)
		}
	}
	log.Printf("LayerNorm small-scale run finished (verify placeholder).")
}
