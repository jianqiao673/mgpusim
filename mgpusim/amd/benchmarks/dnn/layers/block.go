package layers

import (
	"fmt"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// TransformerLayer represents a full Transformer block (Attention + MLP)
type TransformerLayer struct {
	layerIndex int
	to         tensor.Operator

	ln1  *LayerNormLayer
	attn *CausalSelfAttentionLayer
	ln2  *LayerNormLayer
	fc1  *b_FullyConnectedLayer
	gelu *GeluLayer
	fc2  *b_FullyConnectedLayer

	nEmb   int
	nHeads int
	bias   bool
}

// NewTransformerLayer creates a single Transformer block
func NewTransformerLayer(
	index int,
	to tensor.Operator,
	nEmb int,
	nHeads int,
	bias bool,
) *TransformerLayer {
	layer := &TransformerLayer{
		layerIndex: index,
		to:         to,
		nEmb:       nEmb,
		nHeads:     nHeads,
		bias:       bias,
	}

	// Assemble sub-modules
	layer.ln1 = NewLayerNormLayer(
		fmt.Sprintf("ln_1_%d", index), to, nEmb,
	)

	// Attention block with minimum block size 1
	layer.attn = NewCausalSelfAttentionLayer(
		index,
		to,
		CausalSelfAttentionConfig{
			NEmbd:     nEmb,
			NHead:     nHeads,
			Bias:      bias,
			BlockSize: 1,
		},
	)

	layer.ln2 = NewLayerNormLayer(
		fmt.Sprintf("ln_2_%d", index), to, nEmb,
	)
	layer.fc1 = NewBFullyConnectedLayer(
		fmt.Sprintf("fc_1_%d", index), to, nEmb, 4*nEmb, bias,
	)
	layer.gelu = NewGeluLayer()
	layer.fc2 = NewBFullyConnectedLayer(
		fmt.Sprintf("fc_2_%d", index), to, 4*nEmb, nEmb, bias,
	)

	return layer
}

// Forward performs a forward pass through the Transformer block
func (l *TransformerLayer) Forward(x tensor.Tensor) tensor.Tensor {
	// 1. Multi-Head Attention
	attnOut := l.attn.Forward(x)

	// 2. Add & Norm
	residual1 := l.to.ScaleAdd(1.0, 1.0, x, attnOut)
	norm1 := l.ln1.Forward(residual1)

	// 3. Feed-Forward
	fc1Out := l.fc1.Forward(norm1)

	// Reshape to 3D if needed
	shape := fc1Out.Size()
	if len(shape) == 2 {
		batch, hidden := shape[0], shape[1]
		fc1Out = l.to.CreateWithData(fc1Out.Vector(), []int{batch, 1, hidden}, "fc1Out_reshaped")
	} else if len(shape) != 3 {
		panic("expected fc1Out to have 3 dimensions")
	}

	// Flatten for GELU
	fc1Flat := fc1Out.Vector()
	geluFlat := l.gelu.Forward(fc1Flat)

	// Reshape back
	geluOut := l.to.CreateWithData(geluFlat, fc1Out.Size(), "geluOut")

	// Second FC
	fc2Out := l.fc2.Forward(geluOut)

	// 4. Add & Norm
	out := l.to.ScaleAdd(1.0, 1.0, norm1, fc2Out)
	norm2 := l.ln2.Forward(out)

	return norm2
}

// Parameters returns all layer parameters (TODO: implement concatenation)
func (l *TransformerLayer) Parameters() tensor.Tensor {
	return nil
}

// Gradients returns all gradients as a single tensor
func (l *TransformerLayer) Gradients() tensor.Tensor {
	grads := [][]float64{}

	if l.attn != nil {
		for _, g := range l.attn.Gradients() {
			grads = append(grads, g.Vector())
		}
	}

	if l.fc1 != nil {
		grads = append(grads, l.fc1.Gradients().Vector())
	}
	if l.fc2 != nil {
		grads = append(grads, l.fc2.Gradients().Vector())
	}

	// Concatenate into single vector
	totalLen := 0
	for _, v := range grads {
		totalLen += len(v)
	}
	concat := make([]float64, totalLen)
	pos := 0
	for _, v := range grads {
		copy(concat[pos:], v)
		pos += len(v)
	}

	return tensor.NewSimpleTensor([]int{totalLen}, concat, "")
}

// Close releases resources (currently no-op)
func (l *TransformerLayer) Close() {}

// Randomize does nothing (placeholder)
func (l *TransformerLayer) Randomize() {}

// SaveBackward stores forward results for backward pass (placeholder)
func (s *TransformerLayerStack) SaveBackward(input tensor.Tensor) tensor.Tensor {
	return input
}

// LazyRandomize lazily initializes parameters (placeholder)
func (l *TransformerLayer) LazyRandomize() {}
