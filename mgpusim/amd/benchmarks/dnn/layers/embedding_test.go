package layers

import (
	"fmt"
	"math"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
)

func TestEmbeddingLayer(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "EmbeddingLayer Suite")
}

var _ = Describe("EmbeddingLayer", func() {

	var (
		to        *gputensor.GPUOperator
		layerWTE  *EmbeddingLayer
		layerWPE  *EmbeddingLayer
		input     gputensor.Tensor
		output    gputensor.Tensor
		vocabSize int
		embDim    int
	)

	BeforeEach(func() {
		to = &gputensor.GPUOperator{}
		vocabSize = 10
		embDim = 4
		layerWTE = NewEmbeddingLayer("wte", 0, to, vocabSize, embDim)
		layerWPE = NewEmbeddingLayer("wpe", 0, to, vocabSize, embDim)
	})

	// -----------------------
	// Test Random Initialization
	// -----------------------
	It("should initialize token embeddings randomly", func() {
		layerWTE.Randomize()
		weights := layerWTE.GetWeights().Vector()
		Expect(len(weights)).To(Equal(vocabSize * embDim))

		sum := 0.0
		for _, v := range weights {
			sum += math.Abs(v)
		}
		Expect(sum).Should(BeNumerically(">", 0))
		fmt.Println("[Test] Token Embedding Weights:", weights[:8])
	})

	It("should initialize position embeddings with sinusoidal encoding", func() {
		layerWPE.Randomize()
		weights := layerWPE.GetWeights().Vector()
		diff := math.Abs(weights[0] - weights[embDim])
		Expect(diff).Should(BeNumerically(">", 0))
		fmt.Println("[Test] Position Embedding First Position:", weights[:embDim])
		fmt.Println("[Test] Position Embedding Second Position:", weights[embDim:2*embDim])
	})

	// -----------------------
	// Test Forward / Backward
	// -----------------------
	It("should perform forward lookup correctly", func() {
		layerWTE.Randomize()
		input = to.CreateWithData([]float64{
			1, 3, 5,
			2, 4, 6,
		}, []int{2, 3}, "input_idx")

		output = layerWTE.Forward(input)
		Expect(output.Size()).To(Equal([]int{2, 3, embDim}))

		w := layerWTE.GetWeights().Vector()
		idx := int(3)
		start := idx * embDim
		Expect(output.Vector()[embDim:2*embDim]).To(Equal(w[start : start+embDim]))
	})

	It("should produce zero gradients for input indices", func() {
		layerWTE.Randomize()
		input = to.CreateWithData([]float64{
			1, 2,
			3, 4,
		}, []int{2, 2}, "input")

		out := layerWTE.Forward(to.Clone(input))
		gradInput := layerWTE.Backward(out)

		Expect(gradInput.Size()).To(Equal([]int{2, 2}))
		for _, v := range gradInput.Vector() {
			Expect(v).To(Equal(0.0))
		}
	})

	// -----------------------
	// Test SetWeights
	// -----------------------
	It("should allow SetWeights to override embedding values", func() {
		w := make([]float64, vocabSize*embDim)
		for i := range w {
			w[i] = float64(i) / 10.0
		}
		layerWTE.SetWeights(w)
		read := layerWTE.GetWeights().Vector()
		Expect(read).To(Equal(w))
	})

	// -----------------------
	// Test LazyRandomize / SaveForward / SaveBackward
	// -----------------------
	It("should support LazyRandomize and SaveForward / SaveBackward for token embeddings", func() {
		layerWTE.LazyRandomize()
		Expect(len(layerWTE.GetWeights().Vector())).To(Equal(vocabSize * embDim))

		input = to.CreateWithData([]float64{
			0, 1,
			2, 3,
		}, []int{2, 2}, "input_lazy_wte")

		output = layerWTE.SaveForward(input)
		Expect(output.Size()).To(Equal([]int{2, 2, embDim}))

		grad := layerWTE.SaveBackward(output)
		Expect(grad.Size()).To(Equal([]int{2, 2}))

		for _, v := range grad.Vector() {
			Expect(v).To(Equal(0.0))
		}

		sumGrad := 0.0
		for _, g := range layerWTE.weightGradients.Vector() {
			sumGrad += math.Abs(g)
		}
		Expect(sumGrad).Should(BeNumerically(">", 0))
	})

	It("should support LazyRandomize and SaveForward / SaveBackward for position embeddings", func() {
		layerWPE.LazyRandomize()
		Expect(len(layerWPE.GetWeights().Vector())).To(Equal(vocabSize * embDim))

		input = to.CreateWithData([]float64{
			0, 1,
			2, 3,
		}, []int{2, 2}, "input_lazy_wpe")

		output = layerWPE.SaveForward(input)
		Expect(output.Size()).To(Equal([]int{2, 2, embDim}))

		grad := layerWPE.SaveBackward(output)
		Expect(grad.Size()).To(Equal([]int{2, 2}))

		for _, v := range grad.Vector() {
			Expect(v).To(Equal(0.0))
		}
	})
})
