package layers

import (
	"fmt"
	"math"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/akita/v4/sim"
)

func TestGPUEmbeddingLayer(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "GPUEmbeddingLayer Suite")
}

var _ = Describe("GPUEmbeddingLayer", func() {
	var (
		to        tensor.Operator
		layerWTE  *EmbeddingLayer
		layerWPE  *EmbeddingLayer
		input     tensor.Tensor
		output    tensor.Tensor
		vocabSize int
		embDim    int
		gpuDriver *driver.Driver
	)

	BeforeEach(func() {
		gpuDriver = driver.NewDriver()
		engine := sim.NewSerialEngine()
		gpuDriver.Engine = engine

		to = gputensor.NewGPUOperator(gpuDriver, nil)

		// 关键：启动 engine，使 GPU 操作不阻塞
		go engine.Run()

		gpuOp := to.(*gputensor.GPUOperator)
		gpuOp.EnableVerification()

		vocabSize = 10
		embDim = 4
		layerWTE = NewEmbeddingLayer("wte", 0, to, vocabSize, embDim)
		layerWPE = NewEmbeddingLayer("wpe", 0, to, vocabSize, embDim)
	})

	AfterEach(func() {
		// 清理GPU内存
		if layerWTE != nil && layerWTE.parameters != nil {
			to.Free(layerWTE.parameters)
		}
		if layerWTE != nil && layerWTE.gradients != nil {
			to.Free(layerWTE.gradients)
		}
		if layerWPE != nil && layerWPE.parameters != nil {
			to.Free(layerWPE.parameters)
		}
		if layerWPE != nil && layerWPE.gradients != nil {
			to.Free(layerWPE.gradients)
		}
		if input != nil {
			to.Free(input)
		}
		if output != nil {
			to.Free(output)
		}
	})

	It("should initialize weights randomly for token embedding on GPU", func() {
		layerWTE.Randomize()
		weights := layerWTE.GetWeights().Vector()
		Expect(len(weights)).To(Equal(vocabSize * embDim))

		// 检查是否存在非零随机值
		sum := 0.0
		for _, v := range weights {
			sum += math.Abs(v)
		}
		Expect(sum).Should(BeNumerically(">", 0))
		fmt.Println("[GPU Test] Token Embedding Weights:", weights[:8])
	})

	It("should initialize sinusoidal encoding for position embedding on GPU", func() {
		layerWPE.Randomize()
		weights := layerWPE.GetWeights().Vector()

		// 检查位置 0 和 1 是否存在正弦变化
		diff := math.Abs(weights[0] - weights[embDim])
		Expect(diff).Should(BeNumerically(">", 0))
		fmt.Println("[GPU Test] Position Embedding First Position:", weights[:embDim])
		fmt.Println("[GPU Test] Position Embedding Second Position:", weights[embDim:2*embDim])
	})

	It("should perform forward lookup correctly on GPU", func() {
		layerWTE.Randomize()
		input = to.CreateWithData([]float64{
			1, 3, 5,
			2, 4, 6,
		}, []int{2, 3}, "input_idx")

		output = layerWTE.Forward(input)
		defer to.Free(output)
		
		Expect(output.Size()).To(Equal([]int{2, 3, embDim}))
		
		// 验证输出形状和基本数据
		outputVec := output.Vector()
		Expect(len(outputVec)).To(Equal(2 * 3 * embDim))
	})

	It("should produce zero gradients for input (index tensor) on GPU", func() {
		layerWTE.Randomize()
		input = to.CreateWithData([]float64{
			1, 2,
			3, 4,
		}, []int{2, 2}, "input")

		out := layerWTE.Forward(to.Clone(input))
		defer to.Free(out)
		
		gradInput := layerWTE.Backward(out)
		defer to.Free(gradInput)

		Expect(gradInput.Size()).To(Equal([]int{2, 2}))
		
		// 检查梯度是否为零
		gradVec := gradInput.Vector()
		for i, v := range gradVec {
			Expect(v).To(BeNumerically("~", 0.0, 1e-10), 
				"Gradient at index %d should be zero but got %f", i, v)
		}
	})

	It("should support LazyRandomize and SaveForward on GPU", func() {
		layerWTE.LazyRandomize()
		input = to.CreateWithData([]float64{
			0, 1,
			2, 3,
		}, []int{2, 2}, "lazy_input")

		out := layerWTE.SaveForward(to.Clone(input))
		defer to.Free(out)
		
		Expect(out.Size()).To(Equal([]int{2, 2, embDim}))
		
		// 验证输出数据
		outVec := out.Vector()
		Expect(len(outVec)).To(Equal(2 * 2 * embDim))
	})

	It("should handle SaveBackward correctly on GPU (no crash)", func() {
		layerWTE.LazyRandomize()
		input = to.CreateWithData([]float64{
			1, 2,
			3, 4,
		}, []int{2, 2}, "back_input")

		out := layerWTE.SaveForward(to.Clone(input))
		defer to.Free(out)
		
		back := layerWTE.SaveBackward(out)
		defer to.Free(back)

		Expect(back.Size()).To(Equal([]int{2, 2}))
		
		// 检查输出梯度是否为零
		backVec := back.Vector()
		for i, v := range backVec {
			Expect(v).To(BeNumerically("~", 0.0, 1e-10),
				"Backward output at index %d should be zero but got %f", i, v)
		}
	})

	It("should allow SetWeights to override embedding on GPU", func() {
		w := make([]float64, vocabSize*embDim)
		for i := range w {
			w[i] = float64(i) / 10.0
		}
		layerWTE.SetWeights(w)

		read := layerWTE.GetWeights().Vector()
		Expect(len(read)).To(Equal(len(w)))
		
		// 由于浮点数精度问题，使用近似比较
		for i := range w {
			Expect(read[i]).To(BeNumerically("~", w[i], 1e-10),
				"Weight mismatch at index %d: expected %f, got %f", i, w[i], read[i])
		}
	})

	It("should accumulate gradients correctly during backward pass on GPU", func() {
		// 设置确定的权重
		weights := make([]float64, vocabSize*embDim)
		for i := range weights {
			weights[i] = float64(i%embDim) + 1.0 // 简单的模式
		}
		layerWTE.SetWeights(weights)

		// 创建输入
		input = to.CreateWithData([]float64{
			1, 2,
			1, 2, // 重复的token ID
		}, []int{2, 2}, "grad_test_input")

		// 创建梯度输入
		gradInputData := make([]float64, 2*2*embDim)
		for i := range gradInputData {
			gradInputData[i] = 1.0 // 所有梯度都为1
		}
		gradInput := to.CreateWithData(gradInputData, []int{2, 2, embDim}, "grad_input")

		// 执行前向传播保存输入
		_ = layerWTE.Forward(to.Clone(input))
		
		// 执行反向传播
		_ = layerWTE.Backward(gradInput)
		defer to.Free(gradInput)

		// 检查梯度累积
		gradients := layerWTE.Gradients().Vector()
		
		// token 1 和 2 应该累积更多的梯度
		// 因为它们在输入中出现了两次
		token1GradStart := 1 * embDim
		token2GradStart := 2 * embDim
		
		for i := 0; i < embDim; i++ {
			// 每个位置应该累积了2个梯度值（因为出现了两次）
			Expect(gradients[token1GradStart+i]).To(BeNumerically("~", 2.0, 1e-10),
				"Token 1 gradient at dim %d should be 2.0", i)
			Expect(gradients[token2GradStart+i]).To(BeNumerically("~", 2.0, 1e-10),
				"Token 2 gradient at dim %d should be 2.0", i)
		}
	})

	It("should handle out-of-bound token IDs gracefully on GPU", func() {
		layerWTE.Randomize()
		
		// 创建包含超出词汇表范围的token ID的输入
		input = to.CreateWithData([]float64{
			0, float64(vocabSize),    // vocabSize 超出范围
			-1, float64(vocabSize)-1, // -1 超出范围
		}, []int{2, 2}, "oob_input")

		// 应该不会panic
		output = layerWTE.Forward(input)
		defer to.Free(output)
		
		Expect(output.Size()).To(Equal([]int{2, 2, embDim}))
		
		// 检查输出中是否有NaN或Inf
		outputVec := output.Vector()
		for i, v := range outputVec {
			Expect(math.IsNaN(v)).To(BeFalse(), "Output should not contain NaN at index %d", i)
			Expect(math.IsInf(v, 0)).To(BeFalse(), "Output should not contain Inf at index %d", i)
		}
	})

	// 新增测试：验证参数和梯度张量的正确性
	It("should return correct parameters and gradients", func() {
		layerWTE.Randomize()
		
		params := layerWTE.Parameters()
		grads := layerWTE.Gradients()
		
		Expect(params).NotTo(BeNil())
		Expect(grads).NotTo(BeNil())
		Expect(params.Size()[0]).To(Equal(vocabSize * embDim))
		Expect(grads.Size()[0]).To(Equal(vocabSize * embDim))
	})

	// 新增测试：验证输出形状计算
	It("should compute output shape correctly", func() {
		inputShape := []int{4, 8} // batch=4, seq_len=8
		outputShape := layerWTE.GetOutputShape(inputShape)
		
		expectedShape := []int{4, 8, embDim}
		Expect(outputShape).To(Equal(expectedShape))
	})
})