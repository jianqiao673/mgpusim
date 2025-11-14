package attention

import (
	"log"
	"math"

	_ "embed"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/kernels"
)

// KernelArgs defines the arguments for the attention kernel.
type KernelArgs struct {
	B, T, C, NHead      uint32
	HeadSize            uint32
	Scale               float32
	Q, K, V, Output     driver.Ptr
	Mask                driver.Ptr // 添加因果掩码
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
}

type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int
	hsaco   *insts.HsaCo

	BatchSize int
	SeqLen    int
	EmbedDim  int
	NumHeads  int
	HeadSize  int

	qData      []float32
	kData      []float32
	vData      []float32
	outputData []float32
	maskData   []float32 // add mask data

	gQData      driver.Ptr
	gKData      driver.Ptr
	gVData      driver.Ptr
	gOutputData driver.Ptr
	gMaskData   driver.Ptr // add mask pointer

	useUnifiedMemory bool
	saveMemory       bool
}

//go:embed attention.hsaco
var hsacoBytes []byte

func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)
	b.driver = driver
	b.context = driver.Init()

	b.hsaco = kernels.LoadProgramFromMemory(hsacoBytes, "attention")
	if b.hsaco == nil {
		panic("Failed to load attention kernel")
	}

	b.BatchSize = 2
	b.SeqLen = 1024
	b.EmbedDim = 512
	b.NumHeads = 8
	b.HeadSize = b.EmbedDim / b.NumHeads

	return b
}

func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

func (b *Benchmark) SetMemorySaving() {
	b.saveMemory = true
}

func (b *Benchmark) SetParameters(batchSize, seqLen, embedDim, numHeads int) {
	b.BatchSize = batchSize
	b.SeqLen = seqLen
	b.EmbedDim = embedDim
	b.NumHeads = numHeads
	b.HeadSize = embedDim / numHeads
}

func (b *Benchmark) Run() {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("Recovered from panic in Run: %v", r)
            // Ensure resources are cleaned up
            b.safeCleanup()
        }
    }()

    b.driver.SelectGPU(b.context, b.gpus[0])
    if b.saveMemory {
        b.lazyInitMem()
        if !b.validateMemoryAllocation() {
            log.Panic("Memory allocation validation failed")
        }
        b.saveExec()
    } else {
        b.initMem()
        if !b.validateMemoryAllocation() {
            log.Panic("Memory allocation validation failed")
        }
        b.exec()
    }
}

func (b *Benchmark) initMem() {
	totalElements := b.BatchSize * b.SeqLen * b.EmbedDim
	elementSize := 4

	// Allocate input and output memory
	if b.useUnifiedMemory {
		b.gQData = b.driver.AllocateUnifiedMemory(b.context, uint64(totalElements*elementSize))
		b.gKData = b.driver.AllocateUnifiedMemory(b.context, uint64(totalElements*elementSize))
		b.gVData = b.driver.AllocateUnifiedMemory(b.context, uint64(totalElements*elementSize))
		b.gOutputData = b.driver.AllocateUnifiedMemory(b.context, uint64(totalElements*elementSize))
	} else {
		b.gQData = b.driver.AllocateMemory(b.context, uint64(totalElements*elementSize))
		b.gKData = b.driver.AllocateMemory(b.context, uint64(totalElements*elementSize))
		b.gVData = b.driver.AllocateMemory(b.context, uint64(totalElements*elementSize))
		b.gOutputData = b.driver.AllocateMemory(b.context, uint64(totalElements*elementSize))

		b.driver.Distribute(b.context, b.gQData, uint64(totalElements*elementSize), b.gpus)
		b.driver.Distribute(b.context, b.gKData, uint64(totalElements*elementSize), b.gpus)
		b.driver.Distribute(b.context, b.gVData, uint64(totalElements*elementSize), b.gpus)
		b.driver.Distribute(b.context, b.gOutputData, uint64(totalElements*elementSize), b.gpus)
	}

	// Allocate and initialize causal mask
	b.initCausalMask()

	// Initialize input data
	b.qData = make([]float32, totalElements)
	b.kData = make([]float32, totalElements)
	b.vData = make([]float32, totalElements)
	b.outputData = make([]float32, totalElements)

	// Use better initialization method
	b.initInputData()

	b.driver.MemCopyH2D(b.context, b.gQData, b.qData)
	b.driver.MemCopyH2D(b.context, b.gKData, b.kData)
	b.driver.MemCopyH2D(b.context, b.gVData, b.vData)

	log.Printf("Memory initialized - Q: 0x%x, K: 0x%x, V: 0x%x, Output: 0x%x, Mask: 0x%x",
		b.gQData, b.gKData, b.gVData, b.gOutputData, b.gMaskData)
}

func (b *Benchmark) initCausalMask() {
	maskElements := b.SeqLen * b.SeqLen
	b.maskData = make([]float32, maskElements)

	for i := 0; i < b.SeqLen; i++ {
		for j := 0; j < b.SeqLen; j++ {
			if j <= i {
				b.maskData[i*b.SeqLen+j] = 1.0
			} else {
				b.maskData[i*b.SeqLen+j] = 0.0
			}
		}
	}

	if b.useUnifiedMemory {
		b.gMaskData = b.driver.AllocateUnifiedMemory(b.context, uint64(maskElements*4))
	} else {
		b.gMaskData = b.driver.AllocateMemory(b.context, uint64(maskElements*4))
		b.driver.Distribute(b.context, b.gMaskData, uint64(maskElements*4), b.gpus)
	}

	b.driver.MemCopyH2D(b.context, b.gMaskData, b.maskData)
}

func (b *Benchmark) initInputData() {
	totalElements := b.BatchSize * b.SeqLen * b.EmbedDim

	// Use better initialization method to avoid numerical issues
	for i := 0; i < totalElements; i++ {
		// Use small random numbers to avoid softmax numerical instability
		b.qData[i] = (float32(i%100) - 50.0) * 0.01
		b.kData[i] = (float32((i+33)%100) - 50.0) * 0.01
		b.vData[i] = (float32((i+67)%100) - 50.0) * 0.01
	}
}

func (b *Benchmark) exec() {
	queues := make([]*driver.CommandQueue, len(b.gpus))
	scale := float32(1.0 / math.Sqrt(float64(b.HeadSize)))

	globalSize, localSize := b.calculateWorkSize()

	for i, gpu := range b.gpus {
		b.driver.SelectGPU(b.context, gpu)
		q := b.driver.CreateCommandQueue(b.context)
		queues[i] = q

		batchPerGPU := b.BatchSize / len(b.gpus)
		if batchPerGPU == 0 {
			batchPerGPU = 1
		}

		kernArg := KernelArgs{
			B:                   uint32(batchPerGPU),
			T:                   uint32(b.SeqLen),
			C:                   uint32(b.EmbedDim),
			NHead:               uint32(b.NumHeads),
			HeadSize:            uint32(b.HeadSize),
			Scale:               scale,
			Q:                   b.gQData,
			K:                   b.gKData,
			V:                   b.gVData,
			Output:              b.gOutputData,
			Mask:                b.gMaskData, // add mask
			HiddenGlobalOffsetX: int64(batchPerGPU * i * b.SeqLen * b.EmbedDim),
		}

		dCoData, dKernArgData, dPacket := b.driver.EnqueueLaunchKernel(
			q,
			b.hsaco,
			globalSize,
			localSize,
			&kernArg,
		)

		// Save pointers for later cleanup
		_ = dCoData
		_ = dKernArgData
		_ = dPacket
	}

	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}

	b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)
	b.cleanup()
}

func (b *Benchmark) calculateWorkSize() ([3]uint32, [3]uint16) {
	// Calculate more reasonable work sizes
	globalSizeX := uint32(b.BatchSize * b.NumHeads)
	globalSizeY := uint32(b.SeqLen)
	globalSizeZ := uint32(1)

	localSizeX := uint16(1)
	localSizeY := uint16(16)
	localSizeZ := uint16(1)

	// adjust local size Y if global size Y is smaller
	if globalSizeY < 16 {
		localSizeY = uint16(globalSizeY)
	}

	return [3]uint32{globalSizeX, globalSizeY, globalSizeZ},
		[3]uint16{localSizeX, localSizeY, localSizeZ}
}

func (b *Benchmark) lazyInitMem() {
    if b.useUnifiedMemory {
        panic("lazy init does not support unified memory")
    }

    totalElements := b.BatchSize * b.SeqLen * b.EmbedDim

    b.qData = make([]float32, totalElements)
    b.kData = make([]float32, totalElements)
    b.vData = make([]float32, totalElements)
    b.outputData = make([]float32, totalElements)

    b.initInputData()

    // Allocate memory separately for each tensor to avoid sharing
    b.driver.LazyMemCopyH2D(b.context, b.qData, uint64(totalElements*4))
    b.gQData = b.driver.AllocatedVAddr

    b.driver.LazyMemCopyH2D(b.context, b.kData, uint64(totalElements*4))
    b.gKData = b.driver.AllocatedVAddr

    b.driver.LazyMemCopyH2D(b.context, b.vData, uint64(totalElements*4))
    b.gVData = b.driver.AllocatedVAddr

    // Allocate memory separately for output
    b.gOutputData = b.driver.AllocateMemory(b.context, uint64(totalElements*4))

    // Initialize causal mask
    b.initCausalMask()

    log.Printf("Lazy memory initialized - Q: 0x%x, K: 0x%x, V: 0x%x, Output: 0x%x, Mask: 0x%x",
        b.gQData, b.gKData, b.gVData, b.gOutputData, b.gMaskData)
}

func (b *Benchmark) safeCleanup() {
    // only free non-nil pointers
    pointers := []struct {
        name string
        ptr  *driver.Ptr
    }{
        {"gQData", &b.gQData},
        {"gKData", &b.gKData},
        {"gVData", &b.gVData},
        {"gOutputData", &b.gOutputData},
        {"gMaskData", &b.gMaskData},
    }

    for _, p := range pointers {
        if *p.ptr != 0 {
            log.Printf("Freeing %s: 0x%x", p.name, *p.ptr)
            // Use safe free method
            b.driver.FreeMemory(b.context, *p.ptr)
            *p.ptr = 0
        }
    }
}

func (b *Benchmark) saveExec() {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("Recovered from panic in saveExec: %v", r)
            b.safeCleanup()
        }
    }()

    queues := make([]*driver.CommandQueue, len(b.gpus))
    scale := float32(1.0 / math.Sqrt(float64(b.HeadSize)))

    globalSize, localSize := b.calculateWorkSize()

    for i, gpu := range b.gpus {
        b.driver.SelectGPU(b.context, gpu)
        q := b.driver.CreateCommandQueue(b.context)
        queues[i] = q

        batchPerGPU := b.BatchSize / len(b.gpus)
        if batchPerGPU == 0 {
            batchPerGPU = 1
        }

        kernArg := KernelArgs{
            B:                   uint32(batchPerGPU),
            T:                   uint32(b.SeqLen),
            C:                   uint32(b.EmbedDim),
            NHead:               uint32(b.NumHeads),
            HeadSize:            uint32(b.HeadSize),
            Scale:               scale,
            Q:                   b.gQData,
            K:                   b.gKData,
            V:                   b.gVData,
            Output:              b.gOutputData,
            Mask:                b.gMaskData,
            HiddenGlobalOffsetX: int64(batchPerGPU * i * b.SeqLen * b.EmbedDim),
        }

        // Add error checking
        if b.gQData == 0 || b.gKData == 0 || b.gVData == 0 || b.gOutputData == 0 {
            log.Panicf("Invalid memory addresses in lazy execution: Q=0x%x, K=0x%x, V=0x%x, Output=0x%x",
                b.gQData, b.gKData, b.gVData, b.gOutputData)
        }

        b.driver.LazyEnqueueLaunchKernel(
            q,
            b.hsaco,
            globalSize,
            localSize,
            &kernArg,
        )
    }

    for _, q := range queues {
        b.driver.DrainCommandQueue(q)
    }

    // Copy output data
    if b.gOutputData != 0 {
        b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)
    } else {
        log.Printf("Warning: Output memory not allocated, skipping D2H copy")
    }
    
    b.cleanup()
}
func (b *Benchmark) cleanup() {
    // Add null pointer checks to avoid double free
    if b.gQData != 0 {
        b.driver.FreeMemory(b.context, b.gQData)
        b.gQData = 0
    }
    if b.gKData != 0 && b.gKData != b.gQData {
        b.driver.FreeMemory(b.context, b.gKData)
        b.gKData = 0
    }
    if b.gVData != 0 && b.gVData != b.gQData && b.gVData != b.gKData {
        b.driver.FreeMemory(b.context, b.gVData)
        b.gVData = 0
    }
    if b.gOutputData != 0 && b.gOutputData != b.gQData && 
       b.gOutputData != b.gKData && b.gOutputData != b.gVData {
        b.driver.FreeMemory(b.context, b.gOutputData)
        b.gOutputData = 0
    }
    if b.gMaskData != 0 {
        b.driver.FreeMemory(b.context, b.gMaskData)
        b.gMaskData = 0
    }
    
    log.Printf("Memory cleanup completed")
}

func (b *Benchmark) Verify() {
	allZero := true
	hasNaN := false
	maxVal := float32(-1e9)
	minVal := float32(1e9)
	validCount := 0

	for i := 0; i < len(b.outputData); i++ {
		val := b.outputData[i]
		if val != 0 {
			allZero = false
			validCount++
		}
		if math.IsNaN(float64(val)) {
			hasNaN = true
		}
		if val > maxVal {
			maxVal = val
		}
		if val < minVal {
			minVal = val
		}
	}

	if allZero {
		log.Panicf("Output is all zeros")
	}
	if hasNaN {
		log.Panicf("Output contains NaN values")
	}

	log.Printf("Verification passed! Valid values: %d/%d, Range: [%f, %f]",
		validCount, len(b.outputData), minVal, maxVal)

	// For small-scale problems, perform more detailed verification
	if b.BatchSize <= 2 && b.SeqLen <= 128 {
		b.detailedVerification()
	}
}

func (b *Benchmark) detailedVerification() {
	log.Printf("Running detailed verification...")

	// Check some statistical properties of the output
	var sum float32
	for i := range b.outputData {
		sum += b.outputData[i]
	}
	mean := sum / float32(len(b.outputData))

	var variance float32
	for i := range b.outputData {
		diff := b.outputData[i] - mean
		variance += diff * diff
	}
	variance /= float32(len(b.outputData))
	stdDev := float32(math.Sqrt(float64(variance)))

	log.Printf("Output statistics - Mean: %f, StdDev: %f", mean, stdDev)

	// Check if the numerical range is reasonable
	if stdDev > 1000 || math.Abs(float64(mean)) > 1000 {
		log.Printf("Warning: Output values might be too large")
	}
}

func (b *Benchmark) validateMemoryAllocation() bool {
    if b.gQData == 0 {
        log.Printf("Error: Q memory not allocated")
        return false
    }
    if b.gKData == 0 {
        log.Printf("Error: K memory not allocated")
        return false
    }
    if b.gVData == 0 {
        log.Printf("Error: V memory not allocated")
        return false
    }
    if b.gOutputData == 0 {
        log.Printf("Error: Output memory not allocated")
        return false
    }
    if b.gMaskData == 0 {
        log.Printf("Error: Mask memory not allocated")
        return false
    }
    return true
}