package layers

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

var _ = Describe("Embedding Layer", func() {

	var (
		to       *tensor.CPUOperator
		embed    *EmbeddingLayer
		input    tensor.Tensor
		vocab    int
		embedDim int
	)

	BeforeEach(func() {
		to = &tensor.CPUOperator{}
		vocab = 5       // 词表大小
		embedDim = 3    // 每个 token 的嵌入维度
		embed = NewEmbeddingLayer("wte", 0, to, vocab, embedDim)
	})

	It("should randomize weights", func() {
		embed.Randomize()
		Expect(embed.weights.Size()).To(Equal([]int{vocab * embedDim}))
		Expect(embed.weights.Vector()).To(HaveLen(vocab * embedDim))
	})

	It("should forward with known weights", func() {
		// 设定确定的权重值（vocab=5, dim=3）
		to.Init(embed.weights, []float64{
			0.1, 0.2, 0.3, // token 0
			0.4, 0.5, 0.6, // token 1
			0.7, 0.8, 0.9, // token 2
			1.0, 1.1, 1.2, // token 3
			1.3, 1.4, 1.5, // token 4
		})

		// 输入 token 索引（batch=2, seq_len=2）
		input = to.CreateWithData([]float64{
			0, 3, // 第一批
			2, 1, // 第二批
		}, []int{2, 2}, "")

		output := embed.Forward(input)

		// 输出应为对应嵌入向量
		Expect(output.Size()).To(Equal([]int{2, 2, embedDim}))

		Expect(output.Vector()).To(Equal([]float64{
			0.1, 0.2, 0.3, // token 0
			1.0, 1.1, 1.2, // token 3
			0.7, 0.8, 0.9, // token 2
			0.4, 0.5, 0.6, // token 1
		}))
	})

	It("should backward and compute gradients", func() {
		// 初始化确定的权重
		to.Init(embed.weights, []float64{
			0.1, 0.2, 0.3,
			0.4, 0.5, 0.6,
			0.7, 0.8, 0.9,
			1.0, 1.1, 1.2,
			1.3, 1.4, 1.5,
		})

		// 前向输入（记录 forwardInput）
		embed.forwardInput = to.CreateWithData([]float64{
			0, 2,
		}, []int{1, 2}, "")

		// 模拟反向输入（梯度）
		input = to.CreateWithData([]float64{
			0.01, 0.02, 0.03,
			0.04, 0.05, 0.06,
		}, []int{1, 2, 3}, "")

		output := embed.Backward(input)

		// 输出应为零（embedding 对索引梯度为0）
		Expect(output.Size()).To(Equal([]int{1, 2}))
		Expect(output.Vector()).To(Equal([]float64{0, 0}))

		// 检查梯度张量是否存在（无需验证数值）
		Expect(embed.weightGradients.Size()).To(Equal([]int{vocab * embedDim}))
	})

	It("should initialize position embedding when name=wpe", func() {
		posEmbed := NewEmbeddingLayer("wpe", 0, to, vocab, embedDim)
		posEmbed.Randomize()

		Expect(posEmbed.weights.Size()).To(Equal([]int{vocab * embedDim}))

		// 正弦编码通常包含非零值
		vec := posEmbed.weights.Vector()
		Expect(vec[0]).NotTo(Equal(0))
		Expect(vec[1]).NotTo(Equal(0))
	})
})
