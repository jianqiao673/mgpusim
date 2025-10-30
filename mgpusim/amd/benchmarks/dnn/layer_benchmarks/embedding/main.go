// Package embedding implements the embedding algorithm as a benchmark.
package embedding

import (
	"encoding/binary"
	"log"
	"os"
	"unsafe"
	"reflect"
	// embed hsaco files
	_ "embed"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/kernels"
)

// KernelArgs defines kernel arguments for embedding forward
// MUST match the kernel parameter list exactly (order, types, alignment).
type KernelArgs struct {
	Input        driver.Ptr // __global const int* input_indices
	Weight       driver.Ptr // __global const float* weight
	Output       driver.Ptr // __global float* output
	VocabSize    int32      // const int vocab_size
	EmbeddingDim int32      // const int embedding_dim
	BatchSize    int32      // const int batch_size
	SeqLen       int32      // const int seq_len
	PaddingIdx   int32      // const int padding_idx
}

// Benchmark defines a benchmark
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	hsaco   *insts.HsaCo

	VocabSize    int
	EmbeddingDim int
	BatchSize    int
	SeqLen       int

	inputData   []int32
	weightData  []float32
	outputData  []float32
	gInputData  driver.Ptr
	gWeightData driver.Ptr
	gOutputData driver.Ptr

	useUnifiedMemory bool
	saveMemory       bool
}

//go:embed embedding.hsaco
var hsacoBytes []byte

// NewBenchmark returns a benchmark
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)

	b.driver = driver
	b.context = driver.Init()

	// 详细检查HSACO文件（调试信息）
	log.Printf("=== HSACO File Analysis ===")
	log.Printf("File size: %d bytes", len(hsacoBytes))

	// 检查ELF魔数
	if len(hsacoBytes) >= 4 {
		magic := string(hsacoBytes[0:4])
		log.Printf("ELF magic: %x (%s)", hsacoBytes[0:4], magic)
		if magic != "\x7fELF" {
			log.Printf("WARNING: Not a valid ELF file")
		}
	}

	// 检查文件头信息
	if len(hsacoBytes) >= 64 {
		// ELF class (32-bit or 64-bit)
		elfClass := hsacoBytes[4]
		log.Printf("ELF class: %d (1=32-bit, 2=64-bit)", elfClass)

		// ELF data encoding
		dataEnc := hsacoBytes[5]
		log.Printf("Data encoding: %d (1=little endian, 2=big endian)", dataEnc)

		// ELF type
		elfType := binary.LittleEndian.Uint16(hsacoBytes[16:18])
		log.Printf("ELF type: %d", elfType)
	}

	// 尝试加载程序
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PANIC during HSACO loading: %v", r)
			log.Printf("This indicates the HSACO format is not compatible with MGPUSim")
			log.Printf("Possible solutions:")
			log.Printf("1. Recompile the kernel with different target architecture")
			log.Printf("2. Use a different version of MGPUSim")
			log.Printf("3. Check if the HSACO was compiled for the correct GPU architecture")
			os.Exit(1)
		}
	}()

	log.Printf("Attempting to load HSACO (embedding_forward)...")
	b.hsaco = kernels.LoadProgramFromMemory(hsacoBytes, "embedding_forward")
	if b.hsaco == nil {
		// defensive: if load failed, try loading without specific entry (some wrapper libs do this)
		log.Printf("WARNING: kernels.LoadProgramFromMemory returned nil for name 'embedding_forward'")
	}
	log.Printf("HSACO loaded (maybe nil pointer indicates failure handling by kernels.LoadProgramFromMemory).")

	b.gpus = []int{0}

	// 默认参数（可以通过 SetParameters 修改）
	b.VocabSize = 100
	b.EmbeddingDim = 64
	b.BatchSize = 16
	b.SeqLen = 32

	log.Printf("Benchmark initialized successfully")

	return b
}

// SelectGPU selects GPU
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory uses Unified Memory
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// SetMemorySaving sets the memory saving mode
func (b *Benchmark) SetMemorySaving() {
	b.saveMemory = true
}

// SetParameters sets embedding parameters
func (b *Benchmark) SetParameters(vocabSize, embeddingDim, batchSize, seqLen int) {
	b.VocabSize = vocabSize
	b.EmbeddingDim = embeddingDim
	b.BatchSize = batchSize
	b.SeqLen = seqLen
}

// Run runs the benchmark
func (b *Benchmark) Run() {
	if len(b.gpus) == 0 {
		b.gpus = []int{0}
	}
	b.driver.SelectGPU(b.context, b.gpus[0])
	if b.saveMemory {
		b.lazyInitMem()
		b.saveExec()
	} else {
		b.initMem()
		b.exec()
	}
}

func (b *Benchmark) initMem() {
	inputSize := b.BatchSize * b.SeqLen
	weightSize := b.VocabSize * b.EmbeddingDim
	outputSize := b.BatchSize * b.SeqLen * b.EmbeddingDim

	if b.useUnifiedMemory {
		b.gInputData = b.driver.AllocateUnifiedMemory(b.context, uint64(inputSize*4))
		b.gWeightData = b.driver.AllocateUnifiedMemory(b.context, uint64(weightSize*4))
		b.gOutputData = b.driver.AllocateUnifiedMemory(b.context, uint64(outputSize*4))
	} else {
		b.gInputData = b.driver.AllocateMemory(b.context, uint64(inputSize*4))
		b.driver.Distribute(b.context, b.gInputData, uint64(inputSize*4), b.gpus)

		b.gWeightData = b.driver.AllocateMemory(b.context, uint64(weightSize*4))
		b.driver.Distribute(b.context, b.gWeightData, uint64(weightSize*4), b.gpus)

		// Output must be separately allocated — DO NOT reuse weight memory
		b.gOutputData = b.driver.AllocateMemory(b.context, uint64(outputSize*4))
		b.driver.Distribute(b.context, b.gOutputData, uint64(outputSize*4), b.gpus)
	}

	if b.gInputData == 0 || b.gWeightData == 0 || b.gOutputData == 0 {
		log.Fatalf("Memory allocation failed: input=%#x weight=%#x output=%#x",
			b.gInputData, b.gWeightData, b.gOutputData)
	}

	b.inputData = make([]int32, inputSize)
	b.weightData = make([]float32, weightSize)
	b.outputData = make([]float32, outputSize)

	// 初始化输入数据
	for i := 0; i < inputSize; i++ {
		b.inputData[i] = int32(i % b.VocabSize)
	}

	// 初始化权重数据
	for i := 0; i < weightSize; i++ {
		b.weightData[i] = float32(i%10) * 0.1
	}

	b.driver.MemCopyH2D(b.context, b.gInputData, b.inputData)
	b.driver.MemCopyH2D(b.context, b.gWeightData, b.weightData)
	// 输出由 kernel 写回，不需要初始化到 device
}

func (b *Benchmark) exec() {
	queues := make([]*driver.CommandQueue, len(b.gpus))

	var allCoData []driver.Ptr
	var allKernArgData []driver.Ptr
	var allPackets []driver.Ptr

	totalElements := b.BatchSize * b.SeqLen

	// 均匀分配 totalElements 到多个 GPU（base + remainder）
	numGpus := len(b.gpus)
	if numGpus == 0 {
		numGpus = 1
	}
	base := totalElements / numGpus
	rem := totalElements % numGpus
	offsetElements := 0

	for i, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		q := b.driver.CreateCommandQueue(b.context)
		queues[i] = q

		numElements := base
		if i < rem {
			numElements++
		}

		kernArg := KernelArgs{
			Input:        b.gInputData,
			Weight:       b.gWeightData,
			Output:       b.gOutputData,
			VocabSize:    int32(b.VocabSize),
			EmbeddingDim: int32(b.EmbeddingDim),
			BatchSize:    int32(b.BatchSize),
			SeqLen:       int32(b.SeqLen),
			PaddingIdx:   int32(-1),
		}

		// Debug: struct size + expected buffer size
		log.Printf("DEBUG: sizeof(KernelArgs) = %d", unsafe.Sizeof(kernArg))
		expectedOutputBytes := uint64(kernArg.BatchSize) * uint64(kernArg.SeqLen) * uint64(kernArg.EmbeddingDim) * 4
		log.Printf("DEBUG PARAMS: vocab=%d, dim=%d, batch=%d, seq=%d, expected_output_bytes=%d",
			kernArg.VocabSize, kernArg.EmbeddingDim, kernArg.BatchSize, kernArg.SeqLen, expectedOutputBytes)

		// Print each field value via reflection
		rv := reflect.ValueOf(kernArg)
		rt := reflect.TypeOf(kernArg)
		for f := 0; f < rv.NumField(); f++ {
			field := rt.Field(f)
			val := rv.Field(f)
			// safe formatting
			log.Printf("DEBUG FIELD: %s (%v) kind=%v value=%v", field.Name, field.Type, val.Kind(), val.Interface())
		}

		// Sanity check to prevent accidental huge allocations
		const maxAllowed = uint64(1 << 30) // 1 GB; 调试时可以设小些比如 256MB (1<<28)
		if expectedOutputBytes > maxAllowed {
			log.Fatalf("Refusing to launch kernel: expected output bytes too large: %d (> %d). Reduce batch/seq/dim.", expectedOutputBytes, maxAllowed)
		}

		// --- IMPORTANT: pass pointer to struct, exactly what driver expects ---
		log.Printf("DEBUG: launching kernel on gpu=%d numElements=%d", gpu, numElements)
		dCoData, dKernArgData, dPacket := b.driver.EnqueueLaunchKernel(
			q,
			b.hsaco,
			[3]uint32{uint32(numElements), 1, 1},
			[3]uint16{64, 1, 1},
			&kernArg, // <- 必须传指针（driver 通过反射期望 struct pointer）
		)

		log.Printf("DEBUG: launch returned dCoData=0x%x dKernArgData=0x%x dPacket=0x%x", dCoData, dKernArgData, dPacket)

		allCoData = append(allCoData, dCoData)
		allKernArgData = append(allKernArgData, dKernArgData)
		allPackets = append(allPackets, dPacket)

		offsetElements += numElements
	}

	// 等待所有队列完成
	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}

	// 从 device 取回结果
	b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)

	// 清理 GPU 内存
	if b.gInputData != 0 {
		b.driver.FreeMemory(b.context, b.gInputData)
	}
	if b.gWeightData != 0 {
		b.driver.FreeMemory(b.context, b.gWeightData)
	}
	if b.gOutputData != 0 {
		b.driver.FreeMemory(b.context, b.gOutputData)
	}

	for _, ptr := range allCoData {
		if ptr != 0 {
			b.driver.FreeMemory(b.context, ptr)
		}
	}
	for _, ptr := range allKernArgData {
		if ptr != 0 {
			b.driver.FreeMemory(b.context, ptr)
		}
	}
	for _, ptr := range allPackets {
		if ptr != 0 {
			b.driver.FreeMemory(b.context, ptr)
		}
	}
}

// Verify verifies the results
func (b *Benchmark) Verify() {
	tolerance := float32(1e-5)
	errors := 0
	maxErrors := 10

	for i := 0; i < b.BatchSize; i++ {
		for j := 0; j < b.SeqLen; j++ {
			inputIdx := i*b.SeqLen + j
			wordId := b.inputData[inputIdx]

			// padding 没有配置则当作普通索引
			for k := 0; k < b.EmbeddingDim; k++ {
				outputIdx := inputIdx*b.EmbeddingDim + k
				weightIdx := int(wordId)*b.EmbeddingDim + k

				// 边界检查（防止越界）
				var expected float32 = 0
				if wordId >= 0 && int(wordId) < b.VocabSize {
					expected = b.weightData[weightIdx]
				}

				actual := b.outputData[outputIdx]

				if abs(actual-expected) > tolerance {
					if errors < maxErrors {
						log.Printf("Mismatch at batch=%d, seq=%d, dim=%d: expected %f, got %f",
							i, j, k, expected, actual)
					}
					errors++
				}
			}
		}
	}

	if errors > 0 {
		log.Printf("Verification failed with %d errors", errors)
	} else {
		log.Printf("Passed! All results are correct.")
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func (b *Benchmark) lazyInitMem() {
	if b.useUnifiedMemory {
		panic("lazy init does not support unified memory")
	}

	inputSize := b.BatchSize * b.SeqLen
	weightSize := b.VocabSize * b.EmbeddingDim
	outputSize := inputSize * b.EmbeddingDim

	b.inputData = make([]int32, inputSize)
	b.weightData = make([]float32, weightSize)
	b.outputData = make([]float32, outputSize)

	for i := 0; i < inputSize; i++ {
		b.inputData[i] = int32(i % b.VocabSize)
	}
	for i := 0; i < weightSize; i++ {
		b.weightData[i] = float32(i%10) * 0.1
	}

	// Lazy copy inputs/weights to device (driver will set AllocatedVAddr)
	b.driver.LazyMemCopyH2D(b.context, b.inputData, uint64(inputSize*4))
	b.gInputData = b.driver.AllocatedVAddr

	b.driver.LazyMemCopyH2D(b.context, b.weightData, uint64(weightSize*4))
	b.gWeightData = b.driver.AllocatedVAddr

	// 输出单独分配
	b.gOutputData = b.driver.AllocateMemory(b.context, uint64(outputSize*4))
	b.driver.Distribute(b.context, b.gOutputData, uint64(outputSize*4), b.gpus)

	if b.gOutputData == 0 {
		log.Fatalf("lazyInitMem: failed to allocate output memory")
	}
}

func (b *Benchmark) saveExec() {
	queues := make([]*driver.CommandQueue, len(b.gpus))

	var allCoData []driver.Ptr
	var allKernArgData []driver.Ptr
	var allPackets []driver.Ptr

	totalElements := b.BatchSize * b.SeqLen

	// 均匀分配 totalElements 到多个 GPU（base + remainder）
	numGpus := len(b.gpus)
	base := totalElements / numGpus
	rem := totalElements % numGpus

	for i, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		q := b.driver.CreateCommandQueue(b.context)
		queues[i] = q

		numElements := base
		if i < rem {
			numElements++
		}

		kernArg := KernelArgs{
			Input:        b.gInputData,
			Weight:       b.gWeightData,
			Output:       b.gOutputData,
			VocabSize:    int32(b.VocabSize),
			EmbeddingDim: int32(b.EmbeddingDim),
			BatchSize:    int32(b.BatchSize),
			SeqLen:       int32(b.SeqLen),
			PaddingIdx:   int32(-1),
		}

		log.Printf("DEBUG (lazy): sizeof(KernelArgs) = %d", unsafe.Sizeof(kernArg))
		log.Printf("DEBUG PARAMS: vocab=%d, dim=%d, batch=%d, seq=%d",
			kernArg.VocabSize, kernArg.EmbeddingDim, kernArg.BatchSize, kernArg.SeqLen)


		dCoData, dKernArgData, dPacket := b.driver.LazyEnqueueLaunchKernel(
			q,
			b.hsaco,
			[3]uint32{uint32(numElements), 1, 1},
			[3]uint16{64, 1, 1},
			&kernArg,
		)

		allCoData = append(allCoData, dCoData)
		allKernArgData = append(allKernArgData, dKernArgData)
		allPackets = append(allPackets, dPacket)
	}

	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}

	for _, ptr := range allCoData {
		if ptr != 0 {
			b.driver.FreeMemory(b.context, ptr)
		}
	}
	for _, ptr := range allKernArgData {
		if ptr != 0 {
			b.driver.FreeMemory(b.context, ptr)
		}
	}
	for _, ptr := range allPackets {
		if ptr != 0 {
			b.driver.FreeMemory(b.context, ptr)
		}
	}

	b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)

	if b.gInputData != 0 {
		b.driver.FreeMemory(b.context, b.gInputData)
	}
	if b.gWeightData != 0 {
		b.driver.FreeMemory(b.context, b.gWeightData)
	}
	if b.gOutputData != 0 {
		b.driver.FreeMemory(b.context, b.gOutputData)
	}
}
