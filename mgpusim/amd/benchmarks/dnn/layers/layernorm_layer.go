package layers

import (
	"fmt"
	"math"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// LayerNormLayer implements LayerNorm with optional SaveMemory mode
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

	// SaveMemory flag
	saveMemory bool
}

// NewLayerNormLayer creates a LayerNorm layer with optional SaveMemory
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

// SetSaveMemory allows toggling memory-saving mode
func (l *LayerNormLayer) SetSaveMemory(flag bool) {
	l.saveMemory = flag
}

// Randomize initializes gamma=1, beta=0
func (l *LayerNormLayer) Randomize() {
	if l.saveMemory {
		l.LazyRandomize()
		return
	}

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

// LazyRandomize implements memory-saving parameter initialization
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

	fmt.Println("before LazyInitSlices")
	slices := l.to.LazyInitSlices(datas, nums, numParams)
	fmt.Println("after LazyInitSlices")

	l.parameters = slices[0]
	l.gamma = slices[1]
	l.beta = slices[2]
	l.gammaGradient = l.to.Slice(l.gradients, 0, n)
	l.betaGradient = l.to.Slice(l.gradients, n, numParams)
}

// ForwardWithSave dispatches Forward or SaveForward
func (l *LayerNormLayer) ForwardWithSave(input tensor.Tensor) tensor.Tensor {
	if l.saveMemory {
		return l.SaveForward(input)
	}
	return l.Forward(input)
}

// BackwardWithSave dispatches Backward or SaveBackward
func (l *LayerNormLayer) BackwardWithSave(input tensor.Tensor) tensor.Tensor {
	if l.saveMemory {
		return l.SaveBackward(input)
	}
	return l.Backward(input)
}

// Forward implements standard LayerNorm forward
func (l *LayerNormLayer) Forward(input tensor.Tensor) tensor.Tensor {

    l.forwardInput = l.to.Clone(input)
    size := input.Size()
    batch := size[0]
    hidden := l.NEmbd
    out := l.to.Create([]int{batch, hidden})

    // compute mean and variance per batch
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

// Backward implements standard LayerNorm backward
func (l *LayerNormLayer) Backward(input tensor.Tensor) tensor.Tensor {
    defer l.cleanupForwardCache()

    l.to.Clear(l.gradients)
    size := input.Size()
    batch := size[0]
    hidden := l.NEmbd
    outGrad := l.to.Create([]int{batch, hidden})

    inVec := input.Vector()
    normed := l.normalized
    gammaVec := l.gamma.Vector()
    outVec := outGrad.Vector()

    // parameter grads
    for j := 0; j < hidden; j++ {
        var dg, db float64
        for b := 0; b < batch; b++ {
            idx := b*hidden + j
            dg += inVec[idx] * normed[idx]
            db += inVec[idx]
        }
        l.gammaGradient.Vector()[j] = dg
        l.betaGradient.Vector()[j] = db
    }

    // input grads
    for b := 0; b < batch; b++ {
        base := b * hidden
        std := math.Sqrt(l.variance[b] + l.eps)
        var sumDy, sumDyXHat float64
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
            outVec[idx] = (float64(hidden)*dy - sumDy - xHat*sumDyXHat) / float64(hidden) / std
        }
    }

    return outGrad
}

// SaveForward implements memory-saving forward
func (l *LayerNormLayer) SaveForward(input tensor.Tensor) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)
	size := input.Size()
	batch := size[0]
	hidden := l.NEmbd
	out := l.to.Create([]int{batch, hidden})

	inVec := input.Vector()
	outVec := out.Vector()
	gammaVec := l.gamma.Vector()
	betaVec := l.beta.Vector()

	l.normalized = make([]float64, batch*hidden)
	l.mean = make([]float64, batch)
	l.variance = make([]float64, batch)

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

// SaveBackward implements memory-saving backward
func (l *LayerNormLayer) SaveBackward(input tensor.Tensor) tensor.Tensor {
	if input == nil {
        panic("LayerNormLayer.SaveBackward: input tensor is nil")
    }
    if l.forwardInput == nil {
        panic("LayerNormLayer.SaveBackward: forwardInput is nil - was Forward() called?")
    }
    
    l.to.Clear(l.gradients)
    size := input.Size()
    batch := size[0]
    hidden := l.NEmbd
    outGrad := l.to.Create([]int{batch, hidden})

    inVec := input.Vector()
    normed := l.normalized
    gammaVec := l.gamma.Vector()
    outVec := outGrad.Vector()

    // parameter grads
    for j := 0; j < hidden; j++ {
        var dg, db float64
        for b := 0; b < batch; b++ {
            idx := b*hidden + j
            dg += inVec[idx] * normed[idx]
            db += inVec[idx]
        }
        l.gammaGradient.Vector()[j] = dg
        l.betaGradient.Vector()[j] = db
    }

    // input grads
    for b := 0; b < batch; b++ {
        base := b * hidden
        std := math.Sqrt(l.variance[b] + l.eps)
        var sumDy, sumDyXHat float64
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
            outVec[idx] = (float64(hidden)*dy - sumDy - xHat*sumDyXHat) / float64(hidden) / std
        }
    }

    // free saved forward input to save memory
    if l.forwardInput != nil {
        l.to.Free(l.forwardInput)
        l.forwardInput = nil
    }
    
    l.to.Free(input) 
    
    return outGrad
}
// cleanupForwardCache cleans up cached tensors and slices after backward
func (l *LayerNormLayer) cleanupForwardCache() {
    // release forward input tensor
    if l.forwardInput != nil {
        l.to.Free(l.forwardInput)
        l.forwardInput = nil
    }
    
    // release slice caches
    l.normalized = nil
    l.mean = nil
    l.variance = nil
}

// Close releases resources
func (l *LayerNormLayer) Close() {
    l.cleanupForwardCache()
    
    // release parameter and gradient tensors
    if l.parameters != nil {
        l.to.Free(l.parameters)
    }
    if l.gradients != nil {
        l.to.Free(l.gradients)
    }
}

// Accessors
func (l LayerNormLayer) Parameters() tensor.Tensor     { return l.parameters }
func (l LayerNormLayer) Gradients() tensor.Tensor      { return l.gradients }
func (l *LayerNormLayer) GammaGradient() tensor.Tensor { return l.gammaGradient }
func (l *LayerNormLayer) BetaGradient() tensor.Tensor  { return l.betaGradient }
