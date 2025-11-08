package layers

// GeluLayer implements GELU forward/backward using plain float32 slices
type GeluLayer struct {
	forwardIn  []float64
	outputSize int
}

// NewGeluLayer creates a new GELU layer
func NewGeluLayer() *GeluLayer {
	return &GeluLayer{}
}

// Forward performs GELU forward propagation
func (l *GeluLayer) Forward(input []float64) []float64 {
	l.forwardIn = make([]float64, len(input))
	copy(l.forwardIn, input)

	out := make([]float64, len(input))
	const c float64 = 0.79788456 // ≈ sqrt(2/pi)
	for i, x := range input {
		x3 := x * x * x
		out[i] = 0.5 * x * (1 + c*(x+0.044715*x3)) // all float32 now
	}

	l.outputSize = len(out)
	return out
}

// Backward performs backward propagation (placeholder)
func (l *GeluLayer) Backward(input []float64) []float64 {
	return input
}

// SaveForward same as Forward
func (l *GeluLayer) SaveForward(input []float64) []float64 {
	return l.Forward(input)
}

// SaveBackward same as Backward
func (l *GeluLayer) SaveBackward(input []float64) []float64 {
	return l.Backward(input)
}

// Randomize does nothing
func (l *GeluLayer) Randomize() {}

// Parameters returns nil as GELU has no parameters
func (l *GeluLayer) Parameters() []float64 {
	return nil
}

// Gradients returns nil as GELU has no gradients
func (l *GeluLayer) Gradients() []float64 {
	return nil
}
