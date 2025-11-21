package layers

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// TransformerLayerStack represents a stack of Transformer layers
type TransformerLayerStack struct {
	layers []*TransformerLayer
	to     tensor.Operator
	nLayer int
}

// NewTransformerLayerStack creates a stack of Transformer layers
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

// Forward performs forward propagation through each Transformer layer in sequence
func (s *TransformerLayerStack) Forward(x tensor.Tensor) tensor.Tensor {
	out := x
	for i, layer := range s.layers {
		fmt.Printf("  [TransformerStack] Forward layer %d\n", i)
		out = layer.Forward(out)
	}
	return out
}

func (s *TransformerLayerStack) Backward(grad tensor.Tensor) tensor.Tensor {
	
	return grad
}

// Randomize initializes parameters of all layers
func (s *TransformerLayerStack) Randomize() {
	// 	for _, layer := range s.layers {
	// 		layer.Randomize()
	// 	}
}

func (s *TransformerLayerStack) Parameters() tensor.Tensor {
	// TODO: Implement merging logic later
	return nil
}

func (s *TransformerLayerStack) Gradients() tensor.Tensor {
	var allData []float64

	for _, layer := range s.layers {
		g := layer.Gradients()                   // tensor.Tensor
		allData = append(allData, g.Vector()...) // flatten
	}

	// Return a one-dimensional tensor containing gradients of all layers
	return tensor.NewSimpleTensor([]int{len(allData)}, allData, "")
}

// Close releases resources of all layers
func (s *TransformerLayerStack) Close() {
	for _, layer := range s.layers {
		layer.Close()
	}
}
func (s *TransformerLayerStack) LazyRandomize() {

	for _, layer := range s.layers {
		layer.Randomize()
	}
}

func (s *TransformerLayerStack) SaveForward(input tensor.Tensor) tensor.Tensor {

	return input
}
