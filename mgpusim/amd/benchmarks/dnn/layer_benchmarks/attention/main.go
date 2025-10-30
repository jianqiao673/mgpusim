package attention

import (
	"log"
	"math"

	_ "embed"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/kernels"
)

// KernelArgs 修正版
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
	maskData   []float32 // 添加掩码数据

	gQData      driver.Ptr
	gKData      driver.Ptr
	gVData      driver.Ptr
	gOutputData driver.Ptr
	gMaskData   driver.Ptr // 添加掩码指针

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
	totalElements := b.BatchSize * b.SeqLen * b.EmbedDim
	elementSize := 4

	// 分配输入输出内存
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

	// 分配和初始化因果掩码
	b.initCausalMask()

	// 初始化输入数据
	b.qData = make([]float32, totalElements)
	b.kData = make([]float32, totalElements)
	b.vData = make([]float32, totalElements)
	b.outputData = make([]float32, totalElements)

	// 使用更好的初始化方法
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

	// 使用更合理的初始化，避免数值问题
	for i := 0; i < totalElements; i++ {
		// 使用小随机数，避免softmax数值不稳定
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
			Mask:                b.gMaskData, // 添加掩码
			HiddenGlobalOffsetX: int64(batchPerGPU * i * b.SeqLen * b.EmbedDim),
		}

		dCoData, dKernArgData, dPacket := b.driver.EnqueueLaunchKernel(
			q,
			b.hsaco,
			globalSize,
			localSize,
			&kernArg,
		)

		// 保存指针以便后续清理
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
	// 更合理的工作大小计算
	globalSizeX := uint32(b.BatchSize * b.NumHeads)
	globalSizeY := uint32(b.SeqLen)
	globalSizeZ := uint32(1)

	localSizeX := uint16(1)
	localSizeY := uint16(16)
	localSizeZ := uint16(1)

	// 调整局部大小以适应问题规模
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

	// 延迟内存分配和复制
	b.driver.LazyMemCopyH2D(b.context, b.qData, uint64(totalElements*4))
	b.gQData = b.driver.AllocatedVAddr

	b.driver.LazyMemCopyH2D(b.context, b.kData, uint64(totalElements*4))
	b.gKData = b.driver.AllocatedVAddr

	b.driver.LazyMemCopyH2D(b.context, b.vData, uint64(totalElements*4))
	b.gVData = b.driver.AllocatedVAddr

	b.gOutputData = b.gQData

	// 初始化因果掩码
	b.initCausalMask()

	log.Printf("Lazy memory initialized")
}

func (b *Benchmark) saveExec() {
	// 实现与 exec 类似，但使用延迟操作
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

	b.driver.MemCopyD2H(b.context, b.outputData, b.gOutputData)
	b.cleanup()
}

func (b *Benchmark) cleanup() {
	// 清理内存
	b.driver.FreeMemory(b.context, b.gQData)
	b.driver.FreeMemory(b.context, b.gKData)
	b.driver.FreeMemory(b.context, b.gVData)
	b.driver.FreeMemory(b.context, b.gOutputData)
	if b.gMaskData != 0 {
		b.driver.FreeMemory(b.context, b.gMaskData)
	}
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

	// 对于小规模问题，进行更详细的验证
	if b.BatchSize <= 2 && b.SeqLen <= 128 {
		b.detailedVerification()
	}
}

func (b *Benchmark) detailedVerification() {
	log.Printf("Running detailed verification...")

	// 检查输出的一些统计特性
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

	// 检查数值范围是否合理
	if stdDev > 1000 || math.Abs(float64(mean)) > 1000 {
		log.Printf("Warning: Output values might be too large")
	}
}
