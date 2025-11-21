package layers

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// EmbeddingLayer implements a token or positional embedding layer.
// It maps integer token IDs to dense embedding vectors.
type EmbeddingLayer struct {
	Name       string
	layerIndex int
	to         tensor.Operator

	VocabSize    int
	EmbeddingDim int

	parameters      tensor.Tensor
	weights         tensor.Tensor
	gradients       tensor.Tensor
	weightGradients tensor.Tensor
	forwardInput    tensor.Tensor
}

// NewEmbeddingLayer creates a new embedding layer with given vocabulary size and embedding dimension.
func NewEmbeddingLayer(
	name string,
	layerIndex int,
	to tensor.Operator,
	vocabSize, embeddingDim int,
) *EmbeddingLayer {
	numWeight := vocabSize * embeddingDim
	layerIndex = 0 // Embedding layer does not require input gradient computation.

	l := &EmbeddingLayer{
		Name:         name,
		layerIndex:   layerIndex,
		to:           to,
		VocabSize:    vocabSize,
		EmbeddingDim: embeddingDim,
		parameters:   to.Create([]int{numWeight}),
		gradients:    to.Create([]int{numWeight}),
	}

	l.weights = l.parameters
	l.weightGradients = l.gradients

	fmt.Printf("[NewEmbeddingLayer-Allocate] parameters: 0x%x, weights: 0x%x\n",
		l.parameters, l.weights)
	return l
}

func SaveNewEmbeddingLayer(
	name string,
	layerIndex int,
	to tensor.Operator,
	vocabSize, embeddingDim int,
) *EmbeddingLayer {
	numWeight := vocabSize * embeddingDim
	l := &EmbeddingLayer{
		Name:            name,
		layerIndex:      layerIndex,
		to:              to,
		VocabSize:       vocabSize,
		EmbeddingDim:    embeddingDim,
		gradients:       to.Create([]int{numWeight}),
		weightGradients: nil,
	}

	l.weightGradients = l.gradients
	fmt.Printf("[SaveNewEmbeddingLayer-Allocate] gradients: 0x%x, weightGradients: 0x%x\n",
		l.gradients, l.weightGradients)
	return l
}

// Randomize initializes the embedding weights randomly (for token embeddings)
// or using sinusoidal positional encoding (for position embeddings).
func (l *EmbeddingLayer) Randomize() {
	numWeight := l.VocabSize * l.EmbeddingDim
	weights := make([]float64, numWeight)

	if l.Name == "wte" {
		// Token embedding: random initialization
		for i := 0; i < numWeight; i++ {
			weights[i] = (rand.Float64() - 0.5) / float64(l.EmbeddingDim) * 2
		}
		fmt.Println("[EmbeddingLayer.Randomize] Initialized token embedding (wte) randomly")
	} else if l.Name == "wpe" {
		// Positional embedding: sinusoidal encoding
		for pos := 0; pos < l.VocabSize; pos++ {
			for i := 0; i < l.EmbeddingDim; i++ {
				divTerm := math.Exp(-float64(i/2) * math.Log(10000) / float64(l.EmbeddingDim))
				if i%2 == 0 {
					weights[pos*l.EmbeddingDim+i] = math.Sin(float64(pos) * divTerm)
				} else {
					weights[pos*l.EmbeddingDim+i] = math.Cos(float64(pos) * divTerm)
				}
			}
		}
		fmt.Println("[EmbeddingLayer.Randomize] Initialized position embedding (wpe) with sinusoidal encoding")
	}

	l.to.Init(l.weights, weights)
}

// LazyRandomize performs lazy initialization on GPU/CPU operator side.
// Used when delayed memory allocation is required for large-scale training.
func (l *EmbeddingLayer) LazyRandomize() {
	fmt.Printf("[EmbeddingLayer.LazyRandomize] Start lazy initialization for %s\n", l.Name)

	numWeight := l.VocabSize * l.EmbeddingDim
	weights := make([]float64, numWeight)

	if l.Name == "wte" {
		for i := 0; i < numWeight; i++ {
			weights[i] = (rand.Float64() - 0.5) / float64(l.EmbeddingDim) * 2
		}
	} else if l.Name == "wpe" {
		for pos := 0; pos < l.VocabSize; pos++ {
			for i := 0; i < l.EmbeddingDim; i++ {
				divTerm := math.Exp(-float64(i/2) * math.Log(10000) / float64(l.EmbeddingDim))
				if i%2 == 0 {
					weights[pos*l.EmbeddingDim+i] = math.Sin(float64(pos) * divTerm)
				} else {
					weights[pos*l.EmbeddingDim+i] = math.Cos(float64(pos) * divTerm)
				}
			}
		}
	}

	datas := [][]float64{weights}
	nums := []int{numWeight}
	slices := l.to.LazyInitSlices(datas, nums, numWeight)

	l.parameters = slices[0]
	l.weights = slices[0]
	fmt.Printf("[EmbeddingLayer.LazyRandomize] Allocated parameters: 0x%x, weights: 0x%x\n", l.parameters, l.weights)
}

// Forward performs the embedding lookup operation.
// Input: [B, T] integer token indices
// Output: [B, T, C] embedding vectors
func (l *EmbeddingLayer) Forward(input tensor.Tensor) tensor.Tensor {

	l.forwardInput = l.to.Clone(input)
	inputShape := input.Size()
	batch, seq := inputShape[0], inputShape[1]
	output := l.to.Create([]int{batch, seq, l.EmbeddingDim})

	inputVec := input.Vector()
	weightVec := l.weights.Vector()
	outputVec := output.Vector()

	for b := 0; b < batch; b++ {
		for t := 0; t < seq; t++ {
			idx := int(inputVec[b*seq+t])
			if idx < 0 || idx >= l.VocabSize {
				continue
			}
			for d := 0; d < l.EmbeddingDim; d++ {
				outputVec[(b*seq+t)*l.EmbeddingDim+d] = weightVec[idx*l.EmbeddingDim+d]
			}
		}
	}

	return output
}

// Backward accumulates gradients for each embedding vector used in forward pass.
// Input: gradient of output [B, T, C]
// Output: zero tensor [B, T] (no input gradients needed)
func (l *EmbeddingLayer) Backward(input tensor.Tensor) tensor.Tensor {
	defer l.cleanupForwardCache() // 确保缓存被清理

	fmt.Printf("[EmbeddingLayer Backward] VocabSize=%d, EmbeddingDim=%d, weightGradients len=%d\n",
		l.VocabSize, l.EmbeddingDim, len(l.weightGradients.Vector()))

	l.to.Clear(l.gradients)
	gradVec := input.Vector()

	inputShape := l.forwardInput.Size()
	batch, seq := inputShape[0], inputShape[1]
	inputIdxVec := l.forwardInput.Vector()
	gradWeightVec := l.weightGradients.Vector()

	for b := 0; b < batch; b++ {
		for t := 0; t < seq; t++ {
			idx := int(inputIdxVec[b*seq+t])
			if idx < 0 || idx >= l.VocabSize {
				continue
			}
			for d := 0; d < l.EmbeddingDim; d++ {
				gradIndex := (b*seq+t)*l.EmbeddingDim + d
				if gradIndex < len(gradVec) {
					gradWeightVec[idx*l.EmbeddingDim+d] += gradVec[gradIndex]
				}
			}
		}
	}

	outGrad := l.to.Zeros([]int{batch, seq})
	return outGrad
}

// SaveForward performs a memory-optimized forward pass.
// It uses lazy tensor allocation and minimal temporary storage.
func (l *EmbeddingLayer) SaveForward(input tensor.Tensor) tensor.Tensor {
	shape := input.Size()
	if len(shape) == 2 {
		// Original token-id lookup behavior
		batch, seq := shape[0], shape[1]
		fmt.Printf("EmbeddingLayer.SaveForward: input shape: %v, batch=%d, seq=%d (ids)\n", shape, batch, seq)

		// save ids for backward use
		l.forwardInput = l.to.LazyClone(input)

		output := l.to.LazyZeros([]int{batch, seq, l.EmbeddingDim})

		inputVec := input.Vector()
		weightVec := l.weights.Vector()
		outVec := output.Vector()

		for b := 0; b < batch; b++ {
			for t := 0; t < seq; t++ {
				idx := int(inputVec[b*seq+t])
				if idx < 0 || idx >= l.VocabSize {
					continue
				}
				baseOut := (b*seq + t) * l.EmbeddingDim
				baseW := idx * l.EmbeddingDim
				for d := 0; d < l.EmbeddingDim; d++ {
					outVec[baseOut+d] = weightVec[baseW+d]
				}
			}
		}

		// l.to.Free(input)

		fmt.Printf("EmbeddingLayer.SaveForward: output shape: %v\n", output.Size())
		return output

	} else if len(shape) == 3 {
		// Input is already embedding vectors, add positional embeddings for each time step
		batch, seq, emb := shape[0], shape[1], shape[2]
		if emb != l.EmbeddingDim {
			panic(fmt.Sprintf("EmbeddingLayer.SaveForward: 3D input last dim (%d) != EmbeddingDim (%d)", emb, l.EmbeddingDim))
		}
		fmt.Printf("EmbeddingLayer.SaveForward: input shape: %v, treating as pre-embedded -> add positional embeddings\n", shape)

		// Use nil to indicate positional add forward
		l.forwardInput = nil

		// Clone input to avoid modifying original
		output := l.to.LazyClone(input)
		outVec := output.Vector()
		weightVec := l.weights.Vector()

		// Add positional embeddings
		for b := 0; b < batch; b++ {
			for t := 0; t < seq; t++ {
				baseOut := (b*seq + t) * l.EmbeddingDim
				baseW := t * l.EmbeddingDim
				for d := 0; d < l.EmbeddingDim; d++ {
					outVec[baseOut+d] += weightVec[baseW+d]
				}
			}
		}
		return output
	} else {
		panic(fmt.Sprintf("EmbeddingLayer.SaveForward: unsupported input shape: %v", shape))
	}
}

// SaveBackward corresponds to two forward cases
func (l *EmbeddingLayer) SaveBackward(input tensor.Tensor) tensor.Tensor {
	defer l.cleanupForwardCache()

	gradShape := input.Size()
	if len(gradShape) != 3 {
		panic(fmt.Sprintf("EmbeddingLayer.SaveBackward: expected grad 3D [B,T,C], got %v", gradShape))
	}
	batch, seq, emb := gradShape[0], gradShape[1], gradShape[2]
	if emb != l.EmbeddingDim {
		panic("EmbeddingLayer.SaveBackward: grad last dim != EmbeddingDim")
	}

	gradVec := input.Vector()
	gradWeightVec := l.weightGradients.Vector()

	if l.forwardInput == nil {
		// Positional add branch: accumulate gradients by time step
		for b := 0; b < batch; b++ {
			for t := 0; t < seq; t++ {
				baseGrad := (b*seq + t) * l.EmbeddingDim
				baseW := t * l.EmbeddingDim
				for d := 0; d < l.EmbeddingDim; d++ {
					gradWeightVec[baseW+d] += gradVec[baseGrad+d]
				}
			}
		}
	} else {
		// IDs lookup branch: accumulate gradients by token index
		idShape := l.forwardInput.Size()
		if len(idShape) != 2 {
			panic(fmt.Sprintf("EmbeddingLayer.SaveBackward: stored forwardInput shape invalid: %v", idShape))
		}
		idVec := l.forwardInput.Vector()

		for b := 0; b < batch; b++ {
			for t := 0; t < seq; t++ {
				idx := int(idVec[b*seq+t])
				if idx < 0 || idx >= l.VocabSize {
					continue
				}
				baseGrad := (b*seq + t) * l.EmbeddingDim
				baseW := idx * l.EmbeddingDim
				for d := 0; d < l.EmbeddingDim; d++ {
					gradWeightVec[baseW+d] += gradVec[baseGrad+d]
				}
			}
		}
	}

	// Return placeholder grad with same shape as forward
	out := l.to.LazyZeros([]int{batch, seq})

	// l.to.Free(input)

	return out
}

func (l *EmbeddingLayer) cleanupForwardCache() {
	if l.forwardInput != nil {
		l.to.Free(l.forwardInput)
		l.forwardInput = nil
	}
}

func (l *EmbeddingLayer) Close() {
	l.cleanupForwardCache()

	if l.parameters != nil {
		l.to.Free(l.parameters)
	}
	if l.gradients != nil {
		l.to.Free(l.gradients)
	}
}

// Parameters returns the embedding weight tensor.
func (l EmbeddingLayer) Parameters() tensor.Tensor {
	return l.parameters
}

// Gradients returns the gradient tensor associated with embedding weights.
func (l EmbeddingLayer) Gradients() tensor.Tensor {
	return l.gradients
}

// SetWeights loads external weights into the embedding layer (e.g., pretrained embeddings).
func (l *EmbeddingLayer) SetWeights(weights []float64) {
	l.to.Init(l.weights, weights)
}

// GetWeights returns the current embedding weight tensor.
func (l *EmbeddingLayer) GetWeights() tensor.Tensor {
	return l.weights
}

// GetOutputShape returns the output tensor shape given an input shape.
func (l *EmbeddingLayer) GetOutputShape(inputShape []int) []int {
	return []int{inputShape[0], inputShape[1], l.EmbeddingDim}
}
