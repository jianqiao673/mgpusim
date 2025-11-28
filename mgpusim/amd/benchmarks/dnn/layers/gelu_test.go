package layers

import (
	"math"
	"testing"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// GELULayer 简单实现
type GELULayer struct {
	output []float64
}

// 构造函数
func NewGELULayer() *GELULayer {
	return &GELULayer{}
}

// GELU 前向
func (l *GELULayer) Forward(input tensor.Tensor) tensor.Tensor {
	size := input.Size()
	numElem := 1
	for _, s := range size {
		numElem *= s
	}

	out := tensor.NewSimpleTensor(size, make([]float64, numElem), "")
	l.output = make([]float64, numElem)

	inVec := input.Vector()
	outVec := out.Vector()

	for i := 0; i < numElem; i++ {
		x := inVec[i]
		gelu := 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
		l.output[i] = gelu
		outVec[i] = gelu
	}

	return out
}

// 测试 GELU
func TestGELUForward(t *testing.T) {
	data := []float64{-3, -1, 0, 1, 3}
	input := tensor.NewSimpleTensor([]int{1, 5}, data, "")

	layer := NewGELULayer()
	out := layer.Forward(input)

	t.Logf("Input: %v", input.Vector())
	t.Logf("GELU Output: %v", out.Vector())

	// 简单检查值符号和趋势
	for i := 0; i < len(data); i++ {
		if data[i] > 0 && out.Vector()[i] <= 0 {
			t.Errorf("GELU forward failed for input %v", data[i])
		}
		if data[i] < 0 && out.Vector()[i] >= 0 {
			t.Errorf("GELU forward failed for input %v", data[i])
		}
	}
}
