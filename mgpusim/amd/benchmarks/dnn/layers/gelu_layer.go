package layers

import (
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// GeluLayer implements GELU forward/backward with optional SaveMemory mode
type GeluLayer struct {
	to           tensor.Operator
	forwardInput tensor.Tensor
	outputSize   int
	saveMemory   bool
}

// NewGeluLayer creates a GELU layer with optional SaveMemory
func NewGeluLayer(to tensor.Operator, saveMemory bool) *GeluLayer {
	l := &GeluLayer{
		to:         to,
		saveMemory: saveMemory,
	}
	return l
}

// SetSaveMemory allows toggling memory-saving mode
func (l *GeluLayer) SetSaveMemory(flag bool) {
	l.saveMemory = flag
}

// ForwardWithSave dispatches Forward or SaveForward
func (l *GeluLayer) ForwardWithSave(input tensor.Tensor) tensor.Tensor {
	if l.saveMemory {
		return l.SaveForward(input)
	}
	return l.Forward(input)
}

// BackwardWithSave dispatches Backward or SaveBackward
func (l *GeluLayer) BackwardWithSave(input tensor.Tensor) tensor.Tensor {
	if l.saveMemory {
		return l.SaveBackward(input)
	}
	return l.Backward(input)
}

// Forward implements standard GELU forward
func (l *GeluLayer) Forward(input tensor.Tensor) tensor.Tensor {
    l.forwardInput = l.to.Clone(input)
    
    out := l.computeGelu(input)
    
    l.outputSize = len(out.Vector())
    return out
}

func (l *GeluLayer) computeGelu(input tensor.Tensor) tensor.Tensor {
    inVec := input.Vector()
    out := l.to.Create(input.Size())
    outVec := out.Vector()

    const (
        c        = 0.79788456 // sqrt(2/pi)
        cubicCoeff = 0.044715
    )
    
    for i, x := range inVec {
        x3 := x * x * x
        outVec[i] = 0.5 * x * (1 + c*(x+cubicCoeff*x3))
    }
    
    return out
}
// Backward implements standard GELU backward (placeholder)
func (l *GeluLayer) Backward(input tensor.Tensor) tensor.Tensor {
	// GELU placeholder: just return input as gradient
	defer l.cleanupForwardCache()

    return l.computeGeluGradient(input)
}
func (l *GeluLayer) computeGeluGradient(gradOutput tensor.Tensor) tensor.Tensor {
    if l.forwardInput == nil {
        return l.to.Clone(gradOutput)
    }
    
    gradInput := l.to.Clone(gradOutput)
    
    return gradInput
}

func (l *GeluLayer) cleanupForwardCache() {
    if l.forwardInput != nil {
        l.to.Free(l.forwardInput)
        l.forwardInput = nil
    }
}

// SaveForward implements memory-saving GELU forward
func (l *GeluLayer) SaveForward(input tensor.Tensor) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)
	inVec := input.Vector()
	out := l.to.Create(input.Size())
	outVec := out.Vector()

	const c float64 = 0.79788456
	for i, x := range inVec {
		x3 := x * x * x
		outVec[i] = 0.5 * x * (1 + c*(x+0.044715*x3))
	}

	l.outputSize = len(outVec)
	l.to.Free(input)
	return out
}

// SaveBackward implements memory-saving GELU backward (placeholder)
func (l *GeluLayer) SaveBackward(input tensor.Tensor) tensor.Tensor {
	// GELU placeholder: just return input as gradient
	l.to.Free(input)
	return input
}

// Randomize does nothing for GELU
func (l *GeluLayer) Randomize() {}

// Parameters returns nil as GELU has no parameters
func (l *GeluLayer) Parameters() tensor.Tensor { return nil }

// Gradients returns nil as GELU has no gradients
func (l *GeluLayer) Gradients() tensor.Tensor { return nil }
