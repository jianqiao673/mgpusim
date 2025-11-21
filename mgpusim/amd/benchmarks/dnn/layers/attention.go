package layers

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// parameters for CausalSelfAttentionLayer
type CausalSelfAttentionConfig struct {
	NEmbd     int  // embedding dimension
	NHead     int  // number of attention heads
	Bias      bool // whether to use bias
	BlockSize int  // sequence length (for mask)
}

// causal self-attention layer
type CausalSelfAttentionLayer struct {
	layerIndex int
	to         tensor.Operator
	config     CausalSelfAttentionConfig

	// weight parameters
	cAttnWeights tensor.Tensor // [n_embd, 3*n_embd]
	cAttnBias    tensor.Tensor // [3*n_embd] (optional)
	cProjWeights tensor.Tensor // [n_embd, n_embd]
	cProjBias    tensor.Tensor // [n_embd] (optional)

	// gradient tensors
	cAttnWeightsGrad tensor.Tensor
	cAttnBiasGrad    tensor.Tensor
	cProjWeightsGrad tensor.Tensor
	cProjBiasGrad    tensor.Tensor

	// forward propagation cache
	forwardInput     tensor.Tensor
	attentionWeights tensor.Tensor
	softmaxOutput    tensor.Tensor
	qkv              tensor.Tensor
	q, k, v          tensor.Tensor
}

// creates a new causal self-attention layer
func NewCausalSelfAttentionLayer(
	index int,
	to tensor.Operator,
	config CausalSelfAttentionConfig,
) *CausalSelfAttentionLayer {
	if config.NEmbd%config.NHead != 0 {
		panic(fmt.Sprintf("n_embd (%d) must be divisible by n_head (%d)",
			config.NEmbd, config.NHead))
	}

	l := &CausalSelfAttentionLayer{
		layerIndex: index,
		to:         to,
		config:     config,
	}

	// parameter allocation
	l.cAttnWeights = to.Create([]int{config.NEmbd, 3 * config.NEmbd})
	l.cAttnWeightsGrad = to.Create([]int{config.NEmbd, 3 * config.NEmbd})

	if config.Bias {
		l.cAttnBias = to.Create([]int{3 * config.NEmbd})
		l.cAttnBiasGrad = to.Create([]int{3 * config.NEmbd})
	}

	l.cProjWeights = to.Create([]int{config.NEmbd, config.NEmbd})
	l.cProjWeightsGrad = to.Create([]int{config.NEmbd, config.NEmbd})

	if config.Bias {
		l.cProjBias = to.Create([]int{config.NEmbd})
		l.cProjBiasGrad = to.Create([]int{config.NEmbd})
	}

	fmt.Printf("[NewCausalSelfAttentionLayer] layer %d created: n_head=%d, n_embd=%d\n",
		index, config.NHead, config.NEmbd)

	return l
}

// Randomize：randomly initializes parameters
func (l *CausalSelfAttentionLayer) Randomize() {
	xavier := math.Sqrt(2.0 / float64(l.config.NEmbd))

	// c_attn weights
	n1 := l.config.NEmbd * 3 * l.config.NEmbd
	cAttnData := make([]float64, n1)
	for i := range cAttnData {
		cAttnData[i] = (rand.Float64()*2 - 1) * xavier
	}
	l.to.Init(l.cAttnWeights, cAttnData)

	// c_proj weights
	n2 := l.config.NEmbd * l.config.NEmbd
	cProjData := make([]float64, n2)
	for i := range cProjData {
		cProjData[i] = (rand.Float64()*2 - 1) * xavier
	}
	l.to.Init(l.cProjWeights, cProjData)

	// biases
	if l.config.Bias {
		l.to.Init(l.cAttnBias, make([]float64, 3*l.config.NEmbd))
		l.to.Init(l.cProjBias, make([]float64, l.config.NEmbd))
	}
}

// forward propagation
func (l *CausalSelfAttentionLayer) Forward(input tensor.Tensor) tensor.Tensor {
    // chack input dimensions
    inputSize := input.Size()
    if len(inputSize) != 3 {
        panic(fmt.Sprintf("CausalSelfAttentionLayer: expected 3D input, got %dD", len(inputSize)))
    }

    B, T, C := inputSize[0], inputSize[1], inputSize[2]
    if C != l.config.NEmbd {
        panic(fmt.Sprintf("CausalSelfAttentionLayer: input embedding dim %d doesn't match config %d", C, l.config.NEmbd))
    }

    headSize := C / l.config.NHead
    if headSize*l.config.NHead != C {
        panic(fmt.Sprintf("CausalSelfAttentionLayer: embedding dim %d not divisible by n_head %d", C, l.config.NHead))
    }

    // save input for backward
    l.forwardInput = l.to.Clone(input)

    // === 1) QKV projection ===
    in2D := l.to.Reshape(input, []int{B * T, C})
    defer l.to.Free(in2D)

    outputMatrix := l.to.Zeros([]int{B * T, 3 * C})
    defer l.to.Free(outputMatrix)

    // process bias if exists
    if l.config.Bias && l.cAttnBias != nil {
        bias2D := l.to.Reshape(l.cAttnBias, []int{1, 3 * C})
        defer l.to.Free(bias2D)
        
        repeatedBias := l.to.Repeat(bias2D, B*T)
        defer l.to.Free(repeatedBias)
        
        outputMatrix = l.to.ScaleAdd(1.0, 1.0, outputMatrix, repeatedBias)
    }

    qkv2D := l.to.Gemm(false, false, 1.0, 1.0, in2D, l.cAttnWeights, outputMatrix)
    defer l.to.Free(qkv2D)

    l.qkv = l.to.Reshape(qkv2D, []int{B, T, 3 * C})

    // === 2) QKV split ===
    l.q, l.k, l.v = l.splitQKV(l.qkv, B, T, C)

    // === 3) reshape to multi-head ===
    q := l.reshapeToMultiHead(l.q, B, T, headSize)
    defer l.to.Free(q)

    k := l.reshapeToMultiHead(l.k, B, T, headSize)
    defer l.to.Free(k)

    v := l.reshapeToMultiHead(l.v, B, T, headSize)
    defer l.to.Free(v)

    // === 4) compute attention scores ===
    att := l.computeAttentionScores(q, k, B, T, headSize)
    defer l.to.Free(att)

    // === 5) causal mask ===
    attMasked := l.applyCausalMask(att, B, T)
    defer l.to.Free(attMasked)

    // === 6) softmax ===
    attMaskedSize := attMasked.Size()
    totalElements := B * l.config.NHead * T * T
    attMasked2D := l.to.Reshape(attMasked, []int{totalElements / T, T})
    defer l.to.Free(attMasked2D)

    softmaxOut2D := l.to.Softmax(attMasked2D)
    defer l.to.Free(softmaxOut2D)

    softmaxOut := l.to.Reshape(softmaxOut2D, attMaskedSize)
    l.softmaxOutput = l.to.Clone(softmaxOut)
    defer l.to.Free(softmaxOut)

    // === 7) attention-weighted sum ===
    y := l.applyAttention(softmaxOut, v, B, T, headSize)
    defer l.to.Free(y)

    // === 8) reassemble heads ===
    yReassembled := l.reassembleHeads(y, B, T, C)
    defer l.to.Free(yReassembled)

    // === 9) output projection ===
    y2D := l.to.Reshape(yReassembled, []int{B * T, C})
    defer l.to.Free(y2D)

    projOutputMatrix := l.to.Zeros([]int{B * T, C})
    defer l.to.Free(projOutputMatrix)

    if l.config.Bias && l.cProjBias != nil {
        projBias2D := l.to.Reshape(l.cProjBias, []int{1, C})
        defer l.to.Free(projBias2D)
        
        projRepeatedBias := l.to.Repeat(projBias2D, B*T)
        defer l.to.Free(projRepeatedBias)
        
        projOutputMatrix = l.to.ScaleAdd(1.0, 1.0, projOutputMatrix, projRepeatedBias)
    }

    out2D := l.to.Gemm(false, false, 1.0, 1.0, y2D, l.cProjWeights, projOutputMatrix)
    defer l.to.Free(out2D)

    out := l.to.Reshape(out2D, []int{B, T, C})
    return out
}

// splitQKV splits qkv ([B,T,3*C]) into q, k, v ([B,T,C] each)
func (l *CausalSelfAttentionLayer) splitQKV(qkv tensor.Tensor, B, T, C int) (q, k, v tensor.Tensor) {
	totalElements := B * T * 3 * C

	// slice to get q, k, v
	qSlice := l.to.Slice(qkv, 0, B*T*C)
	kSlice := l.to.Slice(qkv, B*T*C, 2*B*T*C)
	vSlice := l.to.Slice(qkv, 2*B*T*C, totalElements)

	// reshape shapes
	q = l.to.Reshape(qSlice, []int{B, T, C})
	k = l.to.Reshape(kSlice, []int{B, T, C})
	v = l.to.Reshape(vSlice, []int{B, T, C})

	// free slices
	l.to.Free(qSlice)
	l.to.Free(kSlice)
	l.to.Free(vSlice)

	return
}

// reshapeToMultiHead：reshape to multi-head format
func (l *CausalSelfAttentionLayer) reshapeToMultiHead(t tensor.Tensor, B, T, headSize int) tensor.Tensor {
	r1 := l.to.Reshape(t, []int{B, T, l.config.NHead, headSize})
	defer l.to.Free(r1)

	out := l.to.Transpose(r1, []int{0, 2, 1, 3})
	return out
}

// computeAttentionScores compute attention scores
func (l *CausalSelfAttentionLayer) computeAttentionScores(q, k tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// transpose k
	kT := l.to.Transpose(k, []int{0, 1, 3, 2})
	defer l.to.Free(kT)

	// batch matrix multiplication
	att := l.batchMatMul(q, kT, B, l.config.NHead, T, headSize, T)

	// scaling
	scale := 1.0 / math.Sqrt(float64(headSize))
	return l.scaleTensor(att, scale)
}

// batchMatMul batch matrix multiplication implementation
func (l *CausalSelfAttentionLayer) batchMatMul(a, b tensor.Tensor, B, nHead, T, M, N int) tensor.Tensor {
	defer func() {
		l.to.Free(a)
		l.to.Free(b)
	}()

	aData := a.Vector()
	bData := b.Vector()
	resultData := make([]float64, B*nHead*T*N)

	// loop over each batch and head
	for bIdx := 0; bIdx < B; bIdx++ {
		for h := 0; h < nHead; h++ {
			// calculate the starting index for the current batch and head
			aBase := (bIdx*nHead + h) * T * M
			bBase := (bIdx*nHead + h) * M * N
			rBase := (bIdx*nHead + h) * T * N

			// calculate matrix multiplication: [T, M] @ [M, N] = [T, N]
			for i := 0; i < T; i++ {
				for j := 0; j < N; j++ {
					sum := 0.0
					for k := 0; k < M; k++ {
						aIdx := aBase + i*M + k
						bIdx := bBase + k*N + j
						sum += aData[aIdx] * bData[bIdx]
					}
					resultData[rBase+i*N+j] = sum
				}
			}
		}
	}

	return l.to.CreateWithData(resultData, []int{B, nHead, T, N}, "batch_matmul")
}

// scaleTensor scale tensor
func (l *CausalSelfAttentionLayer) scaleTensor(t tensor.Tensor, scale float64) tensor.Tensor {
	defer l.to.Free(t)

	data := t.Vector()
	result := make([]float64, len(data))
	for i := range data {
		result[i] = data[i] * scale
	}
	return l.to.CreateWithData(result, t.Size(), "scaled")
}

// applyCausalMask apply causal mask
func (l *CausalSelfAttentionLayer) applyCausalMask(att tensor.Tensor, B, T int) tensor.Tensor {
	defer l.to.Free(att)

	attData := att.Vector()
	result := make([]float64, len(attData))
	copy(result, attData)

	attSize := att.Size()
	nHead := attSize[1]

	// directly apply mask on data to avoid creating full mask matrix
	for b := 0; b < B; b++ {
		for h := 0; h < nHead; h++ {
			for i := 0; i < T; i++ {
				for j := i + 1; j < T; j++ {
					idx := ((b*nHead+h)*T+i)*T + j
					if idx < len(result) {
						result[idx] = -1e9
					}
				}
			}
		}
	}

	return l.to.CreateWithData(result, attSize, "masked_att")
}

// applyAttention apply attention weights
func (l *CausalSelfAttentionLayer) applyAttention(att, v tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// use batch matrix multiplication
	return l.batchMatMul(att, v, B, l.config.NHead, T, T, headSize)
}

// reassembleHeads reshape to multi-head format
func (l *CausalSelfAttentionLayer) reassembleHeads(y tensor.Tensor, B, T, C int) tensor.Tensor {
	defer l.to.Free(y)

	nHead := l.config.NHead
	headDim := C / nHead

	// reshape [B, nHead, T, headDim] -> [B, T, nHead*headDim]
	reshaped := l.to.Reshape(y, []int{B, T, nHead * headDim})
	return reshaped
}

func (l *CausalSelfAttentionLayer) Backward(gradOutput tensor.Tensor) tensor.Tensor {
    defer l.cleanupForwardCache() // clean up forward cache after backward

    B, T, C := gradOutput.Size()[0], gradOutput.Size()[1], gradOutput.Size()[2]

    // initialize cProjWeightsGrad
    if l.cProjWeightsGrad == nil {
        l.cProjWeightsGrad = l.to.Zeros(l.cProjWeights.Size())
    }

    // reshape gradOutput and forwardInput to 2D
    gradOutput2D := l.to.Reshape(gradOutput, []int{B * T, C})
    defer l.to.Free(gradOutput2D)
    
    y2D := l.to.Reshape(l.forwardInput, []int{B * T, C})
    defer l.to.Free(y2D)

    // compute c_proj weights gradient
    l.cProjWeightsGrad = l.to.Gemm(true, false, 1.0, 1.0, y2D, gradOutput2D, l.cProjWeightsGrad)

    // input gradient
    gradInput2D := l.to.Zeros([]int{B * T, C})
    defer l.to.Free(gradInput2D)
    
    gradInput2D = l.to.Gemm(false, true, 1.0, 0.0, gradOutput2D, l.cProjWeights, gradInput2D)
    gradInput := l.to.Reshape(gradInput2D, []int{B, T, C})

    return gradInput
}

// cleanupForwardCache clear forward cache
func (l *CausalSelfAttentionLayer) cleanupForwardCache() {
    tensorsToFree := []*tensor.Tensor{
        &l.forwardInput, &l.softmaxOutput, &l.qkv, &l.q, &l.k, &l.v,
    }
    
    for _, t := range tensorsToFree {
        if *t != nil {
            l.to.Free(*t)
            *t = nil
        }
    }
}

// Parameters return parameters
func (l *CausalSelfAttentionLayer) Parameters() []tensor.Tensor {
	params := []tensor.Tensor{l.cAttnWeights, l.cProjWeights}
	if l.config.Bias {
		params = append(params, l.cAttnBias, l.cProjBias)
	}
	return params
}

// Gradients return gradients
func (l *CausalSelfAttentionLayer) Gradients() []tensor.Tensor {
	grads := []tensor.Tensor{l.cAttnWeightsGrad, l.cProjWeightsGrad}
	if l.config.Bias {
		grads = append(grads, l.cAttnBiasGrad, l.cProjBiasGrad)
	}
	return grads
}

// GetOutputShape get output shape
func (l *CausalSelfAttentionLayer) GetOutputShape(inputShape []int) []int {
	return []int{inputShape[0], inputShape[1], l.config.NEmbd}
}

// SetWeights set weights
func (l *CausalSelfAttentionLayer) SetWeights(cAttnWeights, cProjWeights []float64) {
	l.to.Init(l.cAttnWeights, cAttnWeights)
	l.to.Init(l.cProjWeights, cProjWeights)
}

// SetBiases set biases
func (l *CausalSelfAttentionLayer) SetBiases(cAttnBias, cProjBias []float64) {
	if l.config.Bias {
		l.to.Init(l.cAttnBias, cAttnBias)
		l.to.Init(l.cProjBias, cProjBias)
	}
}

// Close release all resources
func (l *CausalSelfAttentionLayer) Close() {
    l.cleanupForwardCache()
    
    params := l.Parameters()
    grads := l.Gradients()
    
    for _, p := range params {
        if p != nil {
            l.to.Free(p)
        }
    }
    for _, g := range grads {
        if g != nil {
            l.to.Free(g)
        }
    }
}
func (l *CausalSelfAttentionLayer) LazyRandomize() {
	fmt.Printf("[CausalSelfAttentionLayer.LazyRandomize] n_embd=%d, n_head=%d\n",
		l.config.NEmbd, l.config.NHead)

	n1 := l.config.NEmbd * 3 * l.config.NEmbd
	n2 := l.config.NEmbd * l.config.NEmbd
	totalParams := n1 + n2

	xavier := math.Sqrt(2.0 / float64(l.config.NEmbd))

	// === prepare weight data ===
	cAttnData := make([]float64, n1)
	for i := range cAttnData {
		cAttnData[i] = (rand.Float64()*2 - 1) * xavier
	}
	cProjData := make([]float64, n2)
	for i := range cProjData {
		cProjData[i] = (rand.Float64()*2 - 1) * xavier
	}

	nums := []int{n1, n2}

	// === call LazyInitSlices ===
	slices := l.to.LazyInitSlices(
		[][]float64{cAttnData, cProjData},
		nums,
		totalParams,
	)

	// === bind to layer ===
	l.cAttnWeights = slices[0]
	l.cProjWeights = slices[1]

	// === Bias initialization===
	if l.config.Bias {
		l.cAttnBias = l.to.Create([]int{3 * l.config.NEmbd})
		l.cProjBias = l.to.Create([]int{l.config.NEmbd})
		l.to.Init(l.cAttnBias, make([]float64, 3*l.config.NEmbd))
		l.to.Init(l.cProjBias, make([]float64, l.config.NEmbd))
	}
}

// SaveForward memory-optimized forward propagation
func (l *CausalSelfAttentionLayer) SaveForward(input tensor.Tensor) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)

	B, T, C := input.Size()[0], input.Size()[1], input.Size()[2]
	headSize := C / l.config.NHead

	// === 1. QKV ===
	in2D := l.to.LazyReshape(input, []int{B * T, C})
	outputMatrix := l.to.LazyZeros([]int{B * T, 3 * C})

	// Manually implement bias addition: use SaveGemm to add repeated bias
	if l.config.Bias && l.cAttnBias != nil {
		bias2D := l.to.LazyReshape(l.cAttnBias, []int{1, 3 * C})
		repeatedBias := l.to.LazyRepeat(bias2D, B*T)
		// Manual addition: Gemm(scaleA=0, scaleB=1, out=repeatedBias)
		outputMatrix = l.to.SaveGemm(false, false, 0, 1, repeatedBias, repeatedBias, outputMatrix)
	}

	qkv2D := l.to.SaveGemm(false, false, 1.0, 1.0, in2D, l.cAttnWeights, outputMatrix)
	qkv := l.to.LazyReshape(qkv2D, []int{B, T, 3 * C})

	// === 2. Split Q, K, V ===
	q, k, v := l.splitQKV(qkv, B, T, C)
	q = l.reshapeToMultiHead(q, B, T, headSize)
	k = l.reshapeToMultiHead(k, B, T, headSize)
	v = l.reshapeToMultiHead(v, B, T, headSize)

	// === 3. Attention ===
	att := l.computeAttentionScores(q, k, B, T, headSize)
	attMasked := l.applyCausalMask(att, B, T)

	attMasked2D := l.to.LazyReshape(attMasked, []int{B * l.config.NHead * T, T})
	softmaxOut2D := l.to.LazySoftmax(attMasked2D)
	softmaxOut := l.to.LazyReshape(softmaxOut2D, attMasked.Size())

	// === 4. Apply attention ===
	y := l.applyAttention(softmaxOut, v, B, T, headSize)
	y = l.reassembleHeads(y, B, T, C)

	// === 5. Output projection ===
	y2D := l.to.LazyReshape(y, []int{B * T, C})
	projOut := l.to.LazyZeros([]int{B * T, C})

	if l.config.Bias && l.cProjBias != nil {
		projBias2D := l.to.LazyReshape(l.cProjBias, []int{1, C})
		repeatedBias := l.to.LazyRepeat(projBias2D, B*T)
		projOut = l.to.SaveGemm(false, false, 0, 1, repeatedBias, repeatedBias, projOut)
	}

	out2D := l.to.SaveGemm(false, false, 1.0, 1.0, y2D, l.cProjWeights, projOut)
	out := l.to.LazyReshape(out2D, []int{B, T, C})

	// === Free all intermediate variables ===
	for _, t := range []tensor.Tensor{
		in2D, outputMatrix, qkv2D, qkv, q, k, v, att, attMasked,
		softmaxOut2D, softmaxOut, y, y2D, projOut, out2D,
	} {
		if t != nil {
			l.to.Free(t)
		}
	}
	return out
}

// SaveBackward memory-optimized backward propagation
func (l *CausalSelfAttentionLayer) SaveBackward(gradOutput tensor.Tensor) tensor.Tensor {
	B, T, C := gradOutput.Size()[0], gradOutput.Size()[1], gradOutput.Size()[2]

	if l.cProjWeightsGrad == nil {
		l.cProjWeightsGrad = l.to.LazyZeros([]int{C, C})
	}
	l.to.Clear(l.cProjWeightsGrad)

	gradOutput2D := l.to.LazyReshape(gradOutput, []int{B * T, C})
	y2D := l.to.LazyReshape(l.forwardInput, []int{B * T, C})

	// === Weight gradients ===
	l.cProjWeightsGrad = l.to.SaveGemm(true, false, 1.0, 1.0, y2D, gradOutput2D, l.cProjWeightsGrad)

	// === Input gradients ===
	gradInput2D := l.to.LazyZeros([]int{B * T, C})
	gradInput2D = l.to.SaveGemm(false, true, 1.0, 0.0, gradOutput2D, l.cProjWeights, gradInput2D)
	gradInput := l.to.LazyReshape(gradInput2D, []int{B, T, C})

	for _, t := range []tensor.Tensor{gradOutput2D, y2D, gradInput2D} {
		if t != nil {
			l.to.Free(t)
		}
	}
	l.cleanupForwardCache()

	return gradInput
}
