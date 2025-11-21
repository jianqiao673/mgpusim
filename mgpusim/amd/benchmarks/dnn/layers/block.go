package layers

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// TransformerLayer represents a complete Transformer block (Attention + MLP)
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
// NewTransformerLayer creates a single Transformer block (with detailed logging, avoiding BlockSize=0)
func NewTransformerLayer(
	index int,
	to tensor.Operator,
	nEmb int,
	nHeads int,
	bias bool,
) *TransformerLayer {
	fmt.Printf("[NewTransformerLayer] start creating layer %d (n_embd=%d, n_head=%d)\n", index, nEmb, nHeads)

	saveMemory := false

	layer := &TransformerLayer{
		layerIndex: index,
		to:         to,
		nEmb:       nEmb,
		nHeads:     nHeads,
		bias:       bias,
	}

	// === Submodule assembly ===
	fmt.Printf("[NewTransformerLayer] layer %d -> creating ln1\n", index)
	layer.ln1 = NewLayerNormLayer(fmt.Sprintf("ln_1_%d", index), to, nEmb)
	fmt.Printf("[NewTransformerLayer] layer %d -> created ln1\n", index)

	// Note: Do not set BlockSize to 0, set it to at least 1 (or pass as parameter)
	attnBlockSize := 1
	fmt.Printf("[NewTransformerLayer] layer %d -> creating attention (blocksize=%d)\n", index, attnBlockSize)
	layer.attn = NewCausalSelfAttentionLayer(
		index,
		to,
		CausalSelfAttentionConfig{
			NEmbd:     nEmb,
			NHead:     nHeads,
			Bias:      bias,
			BlockSize: attnBlockSize,
		},
	)
	fmt.Printf("[NewTransformerLayer] layer %d -> created attention\n", index)

	fmt.Printf("[NewTransformerLayer] layer %d -> creating ln2\n", index)
	layer.ln2 = NewLayerNormLayer(fmt.Sprintf("ln_2_%d", index), to, nEmb)
	fmt.Printf("[NewTransformerLayer] layer %d -> created ln2\n", index)

	fmt.Printf("[NewTransformerLayer] layer %d -> creating fc1\n", index)
	layer.fc1 = NewBFullyConnectedLayer(fmt.Sprintf("fc_1_%d", index),4, to, nEmb, 4*nEmb, bias)
	fmt.Printf("[NewTransformerLayer] layer %d -> created fc1\n", index)

	fmt.Printf("[NewTransformerLayer] layer %d -> creating gelu\n", index)
	layer.gelu = NewGeluLayer(to, saveMemory)
	fmt.Printf("[NewTransformerLayer] layer %d -> created gelu\n", index)

	fmt.Printf("[NewTransformerLayer] layer %d -> creating fc2\n", index)
	layer.fc2 = NewBFullyConnectedLayer(fmt.Sprintf("fc_2_%d", index),4, to, 4*nEmb, nEmb, bias)
	fmt.Printf("[NewTransformerLayer] layer %d -> created fc2\n", index)

	fmt.Printf("[NewTransformerLayer] layer %d created (n_embd=%d, n_head=%d)\n", index, nEmb, nHeads)
	return layer
}

func (l *TransformerLayer) Forward(x tensor.Tensor) tensor.Tensor {
	fmt.Printf("[Forward] input: type=%T, size=%v, vAddr=%v\n", x, x.Size(), x.Vector())

	// -------------------------
	// 1. Multi-Head Attention
	// -------------------------
	attnOut := l.attn.Forward(x)
	fmt.Printf("[Forward] attnOut: type=%T, size=%v, vAddr=%v\n", attnOut, attnOut.Size(), attnOut.Vector())

	// 2. Add & Norm
	residual1 := l.to.ScaleAdd(1.0, 1.0, x, attnOut)
	fmt.Printf("[Forward] residual1: type=%T, size=%v, vAddr=%v\n", residual1, residual1.Size(), residual1.Vector())

	norm1 := l.ln1.Forward(residual1)
	fmt.Printf("[Forward] norm1: type=%T, size=%v, vAddr=%v\n", norm1, norm1.Size(), norm1.Vector())
	// -------------------------
	// 3. Feed-Forward
	// -------------------------
	fc1Out := l.fc1.Forward(norm1)
	fmt.Printf("[Forward] fc1Out (before reshape): type=%T, size=%v, vAddr=%v\n", fc1Out, fc1Out.Size(), fc1Out.Vector())

	// reshape
	shape := fc1Out.Size()
	if len(shape) == 2 {
		batch, hidden := shape[0], shape[1]
		fc1Out = l.to.CreateWithData(fc1Out.Vector(), []int{batch, 1, hidden}, "fc1Out_reshaped")
		fmt.Printf("[Forward] fc1Out reshaped: type=%T, size=%v, vAddr=%v\n", fc1Out, fc1Out.Size(), fc1Out.Vector())
	} else if len(shape) != 3 {
		panic(fmt.Sprintf("expected fc1Out to have 3 dims, got %v", shape))
	}

	// Call GELU, directly pass tensor.Tensor
	geluOut := l.gelu.ForwardWithSave(fc1Out)
	fmt.Printf("[Forward] geluOut: type=%T, size=%v, vAddr=%v\n", geluOut, geluOut.Size(), geluOut.Vector())

	// If flattening and reshaping is needed (keep original naming)
	geluFlat := geluOut.Vector()
	geluOut = l.to.CreateWithData(geluFlat, fc1Out.Size(), "geluOut")
	fmt.Printf("[Forward] geluOut reshaped: type=%T, size=%v, vAddr=%v\n", geluOut, geluOut.Size(), geluOut.Vector())

	// second FC
	fc2Out := l.fc2.Forward(geluOut)
	fmt.Printf("[Forward] fc2Out: type=%T, size=%v, vAddr=%v\n", fc2Out, fc2Out.Size(), fc2Out.Vector())
	// 4. Add & Norm
	out := l.to.ScaleAdd(1.0, 1.0, norm1, fc2Out)
	fmt.Printf("[Forward] out (before ln2): type=%T, size=%v, vAddr=%v\n", out, out.Size(), out.Vector())

	norm2 := l.ln2.Forward(out)
	fmt.Printf("[Forward] norm2 (output): type=%T, size=%v, vAddr=%v\n", norm2, norm2.Size(), norm2.Vector())

	return norm2
}

// Parameters returns single tensor, complying with layers.Layer interface	
func (l *TransformerLayer) Parameters() tensor.Tensor {
	// TODO: Implement merging logic later
	return nil
}

// TransformerLayer Gradients returns a single tensor, complying with layers.Layer interface
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

	// Concatenate into a single vector
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

	// Use nil as a placeholder for Descriptor
	return tensor.NewSimpleTensor([]int{totalLen}, concat, "")
}

// Close releases resources
func (l *TransformerLayer) Close() {
	// l.attn.Close()
	// l.fc1.Close()
	// l.fc2.Close()
	// l.ln1.Close()
	// l.ln2.Close()
}
func (l *TransformerLayer) Randomize() {}

// SaveBackward saves the forward computation results for use in backpropagation
func (s *TransformerLayerStack) SaveBackward(input tensor.Tensor) tensor.Tensor {
	// If you want to do some caching or record forward outputs, you can implement it here
	// Placeholder for now
	// For example, you can iterate over sublayers and call SaveBackward
	return input
}

// LazyRandomize initializes parameters lazily
func (l *TransformerLayer) LazyRandomize() {
	// Iterate over sublayers and call lazy initialization

}
