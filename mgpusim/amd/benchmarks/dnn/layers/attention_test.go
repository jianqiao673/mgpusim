package layers

import (
	"fmt"
	"math"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// -----------------------------
// 定义测试入口
// -----------------------------
func TestCausalSelfAttentionLayer(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "CausalSelfAttentionLayer Suite")
}

// -----------------------------
// 测试主体
// -----------------------------
var _ = Describe("Causal Self-Attention Layer", func() {
	var (
		to       *tensor.CPUOperator
		attLayer *CausalSelfAttentionLayer
		input    tensor.Tensor
	)

	BeforeEach(func() {
		to = &tensor.CPUOperator{}

		config := CausalSelfAttentionConfig{
			NEmbd:     4,
			NHead:     2,
			Bias:      true,
			BlockSize: 4,
		}

		attLayer = NewCausalSelfAttentionLayer(0, to, config)
		attLayer.Randomize()
	})

	It("should perform forward propagation correctly", func() {
		// 输入张量: [B=1, T=4, C=4]
		inputData := []float64{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6,
		}
		input = to.CreateWithData(inputData, []int{1, 4, 4}, "input")

		output := attLayer.Forward(input)

		Expect(output.Size()).To(Equal([]int{1, 4, 4}))
		Expect(len(output.Vector())).To(Equal(16))
		fmt.Println("[Test] Forward output vector:", output.Vector())

		// 验证输出没有出现 NaN 或 Inf
		for _, v := range output.Vector() {
			Expect(math.IsNaN(v)).To(BeFalse())
			Expect(math.IsInf(v, 0)).To(BeFalse())
		}
	})

	It("should produce deterministic output with same input", func() {
		inputData := []float64{
			0.2, 0.4, 0.6, 0.8,
			0.2, 0.4, 0.6, 0.8,
			0.2, 0.4, 0.6, 0.8,
			0.2, 0.4, 0.6, 0.8,
		}
		input = to.CreateWithData(inputData, []int{1, 4, 4}, "input")

		out1 := attLayer.Forward(input)
		input = to.CreateWithData(inputData, []int{1, 4, 4}, "input2")
		out2 := attLayer.Forward(input)

		Expect(out1.Size()).To(Equal(out2.Size()))
		for i := range out1.Vector() {
			Expect(math.Abs(out1.Vector()[i]-out2.Vector()[i]) < 1e-6).To(BeTrue())
		}
	})

	It("should allow backward computation placeholder", func() {
		inputGrad := to.CreateWithData(make([]float64, 16), []int{1, 4, 4}, "")
		outputGrad := attLayer.Backward(inputGrad)
		Expect(outputGrad.Size()).To(Equal([]int{1, 4, 4}))
	})
})
