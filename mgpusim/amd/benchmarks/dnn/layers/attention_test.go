package layers

import (
	"fmt"
	"math"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

func TestCausalSelfAttentionLayer(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "CausalSelfAttentionLayer Suite")
}

var _ = Describe("CausalSelfAttentionLayer", func() {

	var (
		to     *tensor.CPUOperator
		layer  *CausalSelfAttentionLayer
		config CausalSelfAttentionConfig
		input  tensor.Tensor
		output tensor.Tensor
	)

	BeforeEach(func() {
		to = &tensor.CPUOperator{}
		config = CausalSelfAttentionConfig{
			NHead:     2,
			NEmbd:     8,
			Bias:      true,
			BlockSize: 3,
		}
		layer = NewCausalSelfAttentionLayer(0, to, config)
		layer.Randomize()
	})

	It("should initialize projection weights with correct dimensions", func() {
		Expect(layer.cAttnWeights.Size()).To(Equal([]int{config.NEmbd, 3 * config.NEmbd}))
		Expect(layer.cProjWeights.Size()).To(Equal([]int{config.NEmbd, config.NEmbd}))
		fmt.Println("[Test] c_attn weights:", layer.cAttnWeights.Size())
		fmt.Println("[Test] c_proj weights:", layer.cProjWeights.Size())
	})

	It("should perform standard forward propagation correctly", func() {
		input = to.CreateWithData([]float64{
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
			0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,
			1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
			0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4,
			2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8,
		}, []int{2, 3, 8}, "input")

		output = layer.Forward(input)
		Expect(output.Size()).To(Equal([]int{2, 3, 8}))

		fmt.Println("[Test] Standard Forward output sample:", output.Vector()[:8])
		Expect(math.IsNaN(output.Vector()[0])).To(BeFalse())
		Expect(math.IsInf(output.Vector()[0], 0)).To(BeFalse())
	})

	It("should perform lazy (SaveForward) correctly and produce valid output", func() {
		input = to.CreateWithData([]float64{
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
			0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,
			1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
			0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4,
			2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8,
		}, []int{2, 3, 8}, "input")

		layer.LazyRandomize()
		output = layer.SaveForward(input)

		Expect(output.Size()).To(Equal([]int{2, 3, 8}))
		fmt.Println("[Test] Lazy SaveForward output sample:", output.Vector()[:8])

		for _, v := range output.Vector() {
			Expect(math.IsNaN(v)).To(BeFalse())
			Expect(math.IsInf(v, 0)).To(BeFalse())
		}
	})

	It("should produce deterministic outputs for same input (SaveForward)", func() {
		layer.LazyRandomize()

		input = to.CreateWithData([]float64{
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
			0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,
			1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
			0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4,
			2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8,
		}, []int{2, 3, 8}, "input")

		out1 := layer.SaveForward(to.Clone(input))
		out2 := layer.SaveForward(to.Clone(input))

		Expect(out1.Size()).To(Equal(out2.Size()))

		diff := 0.0
		for i := range out1.Vector() {
			diff += math.Abs(out1.Vector()[i] - out2.Vector()[i])
		}
		fmt.Println("[Test] SaveForward deterministic diff:", diff)
		Expect(diff).Should(BeNumerically("<", 1e-8))
	})

	It("should support SaveBackward without panic", func() {
		defer func() {
			if r := recover(); r != nil {
				Fail(fmt.Sprintf("SaveBackward panicked: %v", r))
			}
		}()

		layer.LazyRandomize()

		input = to.CreateWithData([]float64{
			0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
		}, []int{1, 2, 8}, "input")

		out := layer.SaveForward(to.Clone(input))
		grad := layer.SaveBackward(out)
		Expect(grad.Size()).To(Equal([]int{1, 2, 8}))
		fmt.Println("[Test] SaveBackward finished successfully")
	})
})
