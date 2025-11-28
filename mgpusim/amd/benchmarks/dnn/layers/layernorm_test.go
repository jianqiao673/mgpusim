package layers

import (
	"math"
	"testing"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// 简单 LayerNorm 层，只使用 tensor.Tensor
type LayerNormLayer struct {
	NEmbd      int
	eps        float64
	gamma      tensor.Tensor
	beta       tensor.Tensor
	normalized []float64
	mean       []float64
	variance   []float64
}

// 构造函数
func NewLayerNormLayer(nEmbd int) *LayerNormLayer {
	// gamma=1, beta=0
	gamma := tensor.NewSimpleTensor([]int{nEmbd}, make([]float64, nEmbd), "")
	for i := 0; i < nEmbd; i++ {
		gamma.Vector()[i] = 1.0
	}
	beta := tensor.NewSimpleTensor([]int{nEmbd}, make([]float64, nEmbd), "")
	return &LayerNormLayer{
		NEmbd: nEmbd,
		eps:   1e-5,
		gamma: gamma,
		beta:  beta,
	}
}

// Forward
func (l *LayerNormLayer) Forward(input tensor.Tensor) tensor.Tensor {
	size := input.Size()
	batch := size[0]
	hidden := l.NEmbd
	out := tensor.NewSimpleTensor([]int{batch, hidden}, make([]float64, batch*hidden), "")

	l.normalized = make([]float64, batch*hidden)
	l.mean = make([]float64, batch)
	l.variance = make([]float64, batch)

	inVec := input.Vector()
	outVec := out.Vector()
	gammaVec := l.gamma.Vector()
	betaVec := l.beta.Vector()

	for b := 0; b < batch; b++ {
		base := b * hidden
		var mean float64
		for j := 0; j < hidden; j++ {
			mean += inVec[base+j]
		}
		mean /= float64(hidden)
		l.mean[b] = mean

		var variance float64
		for j := 0; j < hidden; j++ {
			diff := inVec[base+j] - mean
			variance += diff * diff
		}
		variance /= float64(hidden)
		l.variance[b] = variance

		std := math.Sqrt(variance + l.eps)
		for j := 0; j < hidden; j++ {
			idx := base + j
			norm := (inVec[idx] - mean) / std
			l.normalized[idx] = norm
			outVec[idx] = norm*gammaVec[j] + betaVec[j]
		}
	}
	return out
}

// 简单测试
func TestLayerNormForwardBackward(t *testing.T) {
	batch := 2
	nEmbd := 4
	// 输入 Tensor
	data := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	input := tensor.NewSimpleTensor([]int{batch, nEmbd}, data, "")

	layer := NewLayerNormLayer(nEmbd)
	out := layer.Forward(input)

	t.Logf("Input: %v", input.Vector())
	t.Logf("Output: %v", out.Vector())

	// 验证均值约为0，方差约为1
	for b := 0; b < batch; b++ {
		base := b * nEmbd
		var mean float64
		var variance float64
		for j := 0; j < nEmbd; j++ {
			x := layer.normalized[base+j]
			mean += x
			variance += x * x
		}
		mean /= float64(nEmbd)
		variance /= float64(nEmbd)
		if math.Abs(mean) > 1e-6 {
			t.Errorf("batch %d: normalized mean=%v not ~0", b, mean)
		}
		if math.Abs(variance-1.0) > 1e-3 {
			t.Errorf("batch %d: normalized variance=%v not ~1", b, variance)
		}
	}
}
