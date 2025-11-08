package layers

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// TransformerLayerStack 表示多个 Transformer 层的堆叠（例如 GPT 的 12/24 层）
type TransformerLayerStack struct {
	layers []*TransformerLayer
	to     tensor.Operator
	nLayer int
}

// NewTransformerLayerStack 创建 Transformer 层堆叠
func NewTransformerLayerStack(
	to tensor.Operator,
	nLayer int,
	nEmb int,
	nHeads int,
	bias bool,
) *TransformerLayerStack {
	stack := &TransformerLayerStack{
		to:     to,
		nLayer: nLayer,
	}

	for i := 0; i < nLayer; i++ {
		layer := NewTransformerLayer(i, to, nEmb, nHeads, bias)
		stack.layers = append(stack.layers, layer)
	}

	fmt.Printf("[NewTransformerLayerStack] created with %d layers\n", nLayer)
	return stack
}

// Forward 前向传播：依次通过每个 Transformer 层
func (s *TransformerLayerStack) Forward(x tensor.Tensor) tensor.Tensor {
	out := x
	for i, layer := range s.layers {
		fmt.Printf("  [TransformerStack] Forward layer %d\n", i)
		out = layer.Forward(out)
	}
	return out
}

func (s *TransformerLayerStack) Backward(grad tensor.Tensor) tensor.Tensor {
	// out := grad
	// // 逆序调用每个 TransformerLayer 的 Backward
	// for i := len(s.layers) - 1; i >= 0; i-- {
	// 	out = s.layers[i].Backward(out)
	// }
	return grad
}

// Randomize 初始化所有层参数
func (s *TransformerLayerStack) Randomize() {
	// 	for _, layer := range s.layers {
	// 		layer.Randomize()
	// 	}
}

func (s *TransformerLayerStack) Parameters() tensor.Tensor {
	// TODO: 后续实现真正的合并逻辑
	return nil
}

func (s *TransformerLayerStack) Gradients() tensor.Tensor {
	var allData []float64

	for _, layer := range s.layers {
		g := layer.Gradients()                   // tensor.Tensor
		allData = append(allData, g.Vector()...) // flatten
	}

	// 返回一个一维 tensor，包含所有层梯度
	return tensor.NewSimpleTensor([]int{len(allData)}, allData, "")
}

// Close 释放所有层资源
func (s *TransformerLayerStack) Close() {
	for _, layer := range s.layers {
		layer.Close()
	}
}
func (s *TransformerLayerStack) LazyRandomize() {
	// 如果你还没有实现逻辑，可以先留空
	// 或者遍历每一层调用 Randomize
	for _, layer := range s.layers {
		layer.Randomize()
	}
}

func (s *TransformerLayerStack) SaveForward(input tensor.Tensor) tensor.Tensor {
	// 占位实现，暂时不保存任何内容
	return input
}
