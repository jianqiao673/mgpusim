package layers

import (
	"fmt"
	"math"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// LayerNormLayer matches the coding style and interfaces of FullyConnectedLayer.
type LayerNormLayer struct {
	layerIndex int
	to         tensor.Operator

	NEmbd int
	eps   float64

	parameters tensor.Tensor // [gamma, beta]
	gradients  tensor.Tensor // [dgamma, dbeta]

	gamma         tensor.Tensor
	beta          tensor.Tensor
	gammaGradient tensor.Tensor
	betaGradient  tensor.Tensor

	// saved for backward
	forwardInput tensor.Tensor
	normalized   []float64 // cached normalized values (batch * NEmbd)
	mean         []float64 // per-batch mean
	variance     []float64 // per-batch variance
}

// NewLayerNormLayer creates a LayerNorm layer.
// Signature style same as FullyConnectedLayer's New... functions.
// NewLayerNormLayer creates a LayerNorm layer.
// Signature style same as FullyConnectedLayer's New... functions.
func NewLayerNormLayer(index interface{}, to tensor.Operator, nEmbd int) *LayerNormLayer {
	var idx int
	switch v := index.(type) {
	case int:
		idx = v
	case string:
		// if a string like "ln_f" is given, ignore index value
		fmt.Printf("[NewLayerNormLayer] layer name: %s (auto index=0)\n", v)
		idx = 0
	default:
		idx = 0
	}

	numParams := nEmbd * 2
	l := &LayerNormLayer{
		layerIndex: idx,
		to:         to,
		NEmbd:      nEmbd,
		eps:        1e-5,
		parameters: to.Create([]int{numParams}),
		gradients:  to.Create([]int{numParams}),
	}

	l.gamma = to.Slice(l.parameters, 0, nEmbd)
	l.beta = to.Slice(l.parameters, nEmbd, numParams)
	l.gammaGradient = to.Slice(l.gradients, 0, nEmbd)
	l.betaGradient = to.Slice(l.gradients, nEmbd, numParams)

	fmt.Printf("[NewLayerNormLayer-Allocate] parameters: 0x%x, gamma: 0x%x, beta: 0x%x\n",
		l.parameters, l.gamma, l.beta)
	fmt.Printf("[NewLayerNormLayer-Allocate] gradients: 0x%x, gammaGrad: 0x%x, betaGrad: 0x%x\n",
		l.gradients, l.gammaGradient, l.betaGradient)

	return l
}

// Randomize sets gamma=1, beta=0
func (l *LayerNormLayer) Randomize() {
	n := l.NEmbd
	g := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		g[i] = 1.0
		b[i] = 0.0
	}
	l.to.Init(l.gamma, g)
	l.to.Init(l.beta, b)
}

// LazyRandomize mirrors FullyConnectedLayer.LazyRandomize style.
func (l *LayerNormLayer) LazyRandomize() {
	fmt.Printf("LayerNormLayer.LazyRandomize\n")

	n := l.NEmbd
	g := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		g[i] = 1.0
		b[i] = 0.0
	}
	numParams := n * 2
	datas := [][]float64{g, b}
	nums := []int{n, numParams}

	slices := l.to.LazyInitSlices(datas, nums, numParams)
	// Following same layout as FullyConnectedLayer.LazyRandomize
	l.parameters = slices[0]
	l.gamma = slices[1]
	l.beta = slices[2]

	fmt.Printf("[LazyRandomize-Allocate] parameters: 0x%x, gamma: 0x%x, beta: 0x%x\n",
		l.parameters, l.gamma, l.beta)
}

// Forward: input shape [batch, NEmbd] (same assumptions as FC Forward)
func (l *LayerNormLayer) Forward(input tensor.Tensor) tensor.Tensor {
	// keep clone for backward
	l.forwardInput = l.to.Clone(input)

	size := input.Size()
	batch := size[0]
	hidden := l.NEmbd

	// allocate output
	out := l.to.Create([]int{batch, hidden})

	// prepare caches
	l.normalized = make([]float64, batch*hidden)
	l.mean = make([]float64, batch)
	l.variance = make([]float64, batch)

	inVec := input.Vector()
	outVec := out.Vector()
	gammaVec := l.gamma.Vector()
	betaVec := l.beta.Vector()

	for b := 0; b < batch; b++ {
		// compute mean
		var mean float64
		base := b * hidden
		for j := 0; j < hidden; j++ {
			mean += inVec[base+j]
		}
		mean /= float64(hidden)
		l.mean[b] = mean

		// compute variance
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

	// free input as other layers do
	l.to.Free(input)
	return out
}

// Backward: input is gradOutput with shape [batch, NEmbd]
func (l *LayerNormLayer) Backward(input tensor.Tensor) tensor.Tensor {
	// clear gradients buffer (match FC style)
	l.to.Clear(l.gradients)

	size := input.Size()
	batch := size[0]
	hidden := l.NEmbd

	// allocate output grad
	outGrad := l.to.Create([]int{batch, hidden})

	// ensure gradient parameter tensors exist (they are slices of l.gradients)
	// compute dgamma and dbeta by summation
	inVec := input.Vector()
	normed := l.normalized
	gammaVec := l.gamma.Vector()

	// zero gradients storage (to be safe)
	// l.to.Clear(l.gradients) already called

	// compute gammaGrad and betaGrad
	for j := 0; j < hidden; j++ {
		var dg float64
		var db float64
		for b := 0; b < batch; b++ {
			idx := b*hidden + j
			dg += inVec[idx] * normed[idx]
			db += inVec[idx]
		}
		l.gammaGradient.Vector()[j] = dg
		l.betaGradient.Vector()[j] = db
	}

	// compute dx
	outVec := outGrad.Vector()
	for b := 0; b < batch; b++ {
		base := b * hidden
		//mean := l.mean[b]
		variance := l.variance[b]
		std := math.Sqrt(variance + l.eps)

		// intermediate sums
		var sumDy float64
		var sumDyXHat float64
		for j := 0; j < hidden; j++ {
			idx := base + j
			dy := inVec[idx] * gammaVec[j]
			sumDy += dy
			sumDyXHat += dy * normed[idx]
		}

		for j := 0; j < hidden; j++ {
			idx := base + j
			dy := inVec[idx] * gammaVec[j]
			xHat := normed[idx]
			// dx = (1/N) * (1/std) * (N*dy - sumDy - xHat * sumDyXHat)
			val := (float64(hidden)*dy - sumDy - xHat*sumDyXHat) / float64(hidden)
			val = val / std
			outVec[idx] = val
		}
	}

	// free saved inputs like other layers
	l.to.Free(l.forwardInput)
	l.to.Free(input)

	return outGrad
}

// Parameters returns parameters tensor [gamma,beta]
func (l LayerNormLayer) Parameters() tensor.Tensor {
	return l.parameters
}

// Gradients returns gradients tensor [dgamma, dbeta]
func (l LayerNormLayer) Gradients() tensor.Tensor {
	return l.gradients
}

// SaveForward: memory-saving variant using Lazy* calls (mirror FC style)
func (l *LayerNormLayer) SaveForward(input tensor.Tensor) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)

	size := input.Size()
	batch := size[0]
	hidden := l.NEmbd

	// NOTE: CPUOperator's Lazy* are no-op, but we keep calls to match style.
	out := l.to.Create([]int{batch, hidden})

	// We'll compute numerically here as in Forward (can't rely on Lazy ops exist)
	inVec := input.Vector()
	outVec := out.Vector()

	// caches
	l.normalized = make([]float64, batch*hidden)
	l.mean = make([]float64, batch)
	l.variance = make([]float64, batch)
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

	l.to.Free(input)
	return out
}

// SaveBackward: memory-saving backward; mirror SaveForward
func (l *LayerNormLayer) SaveBackward(input tensor.Tensor) tensor.Tensor {
	// same as Backward but we call Lazy* copies where appropriate (kept minimal)
	l.to.Clear(l.gradients)

	size := input.Size()
	batch := size[0]
	hidden := l.NEmbd

	outGrad := l.to.Create([]int{batch, hidden})

	inVec := input.Vector()
	normed := l.normalized
	gammaVec := l.gamma.Vector()

	// compute parameter grads
	for j := 0; j < hidden; j++ {
		var dg float64
		var db float64
		for b := 0; b < batch; b++ {
			idx := b*hidden + j
			dg += inVec[idx] * normed[idx]
			db += inVec[idx]
		}
		l.gammaGradient.Vector()[j] = dg
		l.betaGradient.Vector()[j] = db
	}

	// compute input grads
	outVec := outGrad.Vector()
	for b := 0; b < batch; b++ {
		base := b * hidden
		std := math.Sqrt(l.variance[b] + l.eps)

		var sumDy float64
		var sumDyXHat float64
		for j := 0; j < hidden; j++ {
			idx := base + j
			dy := inVec[idx] * gammaVec[j]
			sumDy += dy
			sumDyXHat += dy * normed[idx]
		}

		for j := 0; j < hidden; j++ {
			idx := base + j
			dy := inVec[idx] * gammaVec[j]
			xHat := normed[idx]
			val := (float64(hidden)*dy - sumDy - xHat*sumDyXHat) / float64(hidden)
			outVec[idx] = val / std
		}
	}

	l.to.Free(l.forwardInput)
	l.to.Free(input)
	return outGrad
}

func (l *LayerNormLayer) GammaGradient() tensor.Tensor {
	return l.gammaGradient
}

func (l *LayerNormLayer) BetaGradient() tensor.Tensor {
	return l.betaGradient
}
