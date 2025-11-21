package layers

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// FullyConnectedLayer implements a fully connected layer compatible with MiniGPT.
type b_FullyConnectedLayer struct {
	Name       string
	LayerIndex int
	to         tensor.Operator

	InputSize  int
	OutputSize int

	parameters      tensor.Tensor
	weights         tensor.Tensor
	bias            tensor.Tensor
	gradients       tensor.Tensor
	weightGradients tensor.Tensor
	biasGradients   tensor.Tensor
	forwardInput    tensor.Tensor

	UseBias bool
}

// NewBFullyConnectedLayer creates a fully connected layer.
func NewBFullyConnectedLayer(
	name string,
	layerIndex int,
	to tensor.Operator,
	inputSize, outputSize int,
	useBias bool,
) *b_FullyConnectedLayer {

	numWeight := inputSize * outputSize
	numBias := 0
	if useBias {
		numBias = outputSize
	}
	numParams := numWeight + numBias

	l := &b_FullyConnectedLayer{
		Name:       name,
		to:         to,
		InputSize:  inputSize,
		OutputSize: outputSize,
		UseBias:    useBias,
	}

	fmt.Printf("[BFCLayer:%s] >>> Allocating parameters (input=%d, output=%d, useBias=%v)\n",
		l.Name, inputSize, outputSize, useBias)

	// Allocate parameters
	l.parameters = to.Create([]int{numParams})
	l.weights = to.Slice(l.parameters, 0, numWeight)
	if useBias {
		l.bias = to.Slice(l.parameters, numWeight, numParams)
	}

	// Allocate gradients
	l.gradients = to.Create([]int{numParams})
	l.weightGradients = to.Slice(l.gradients, 0, numWeight)
	if useBias {
		l.biasGradients = to.Slice(l.gradients, numWeight, numParams)
	}

	fmt.Printf("[BFCLayer:%s]  All inits done\n", l.Name)

	return l
}

func (l *b_FullyConnectedLayer) LazyRandomize() {
	fmt.Printf("FullyConnectedLayer.LazyRandomize, useBias: %v\n", l.UseBias)

	numWeight := l.InputSize * l.OutputSize

	weights := make([]float64, numWeight)
	for i := 0; i < numWeight; i++ {
		weights[i] = (rand.Float64() - 0.5) / float64(l.InputSize) * 2
	}

	// Prepare data for lazy initialization
	var datas [][]float64
	var nums []int

	// First data block: weights
	datas = append(datas, weights)
	nums = append(nums, numWeight)

	if l.UseBias {
		numBias := l.OutputSize
		bias := make([]float64, numBias)
		for i := 0; i < numBias; i++ {
			bias[i] = rand.Float64()*2 - 1
		}
		datas = append(datas, bias)
		nums = append(nums, numWeight) // Bias starts after weights
	}

	numParams := numWeight
	if l.UseBias {
		numParams += l.OutputSize
	}

	slices := l.to.LazyInitSlices(datas, nums, numParams)
	l.parameters = slices[0]
	l.weights = slices[1]
	if l.UseBias {
		l.bias = slices[2]
	}
}

// Optimized SaveBackward version
func (l *b_FullyConnectedLayer) SaveBackward(input tensor.Tensor) tensor.Tensor {
	defer l.cleanupForwardCache() // Clear cache immediately after computation
	defer l.to.Free(input)        // Input gradients no longer needed

	l.to.Clear(l.gradients)

	l.saveCalculateWeightGradients(input)
	if l.UseBias {
		l.lazyCalculateBiasGradients(input)
	}

	var output tensor.Tensor
	if l.LayerIndex > 0 {
		output = l.saveCalculateInputGradients(input)
	}

	return output
}

// Optimized SaveForward version
func (l *b_FullyConnectedLayer) SaveForward(input tensor.Tensor) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)

	in := l.to.LazyReshape(input, []int{input.Size()[0], l.InputSize})
	defer l.to.Free(in)

	weightMat := l.to.LazyReshape(l.weights, []int{l.InputSize, l.OutputSize})
	defer l.to.Free(weightMat)

	var out tensor.Tensor

	if l.UseBias {
		// With bias using lazy operations
		biasMat := l.to.LazyRepeat(l.bias, input.Size()[0])
		defer l.to.Free(biasMat)

		biasMatReshape := l.to.LazyReshape(biasMat, []int{input.Size()[0], l.OutputSize})
		defer l.to.Free(biasMatReshape)

		out = l.to.SaveGemm(false, false, 1, 1, in, weightMat, biasMatReshape)
	} else {
		// Without bias
		zeroBias := l.to.LazyZeros([]int{input.Size()[0], l.OutputSize})
		defer l.to.Free(zeroBias)

		out = l.to.SaveGemm(false, false, 1, 1, in, weightMat, zeroBias)
	}

	l.to.Free(input) // Original input no longer needed

	return out
}

// Randomize initializes the weights using Xavier initialization
func (l *b_FullyConnectedLayer) Randomize() {
	numWeight := l.InputSize * l.OutputSize
	limit := math.Sqrt(6.0 / float64(l.InputSize+l.OutputSize))
	weights := make([]float64, numWeight)
	for i := 0; i < numWeight; i++ {
		weights[i] = rand.Float64()*2*limit - limit
	}
	l.to.Init(l.weights, weights)

	if l.UseBias {
		bias := make([]float64, l.OutputSize)
		l.to.Init(l.bias, bias)
	}

	fmt.Printf("[BFCLayer:%s] Weights randomized (Xavier init, limit=%.4f)\n", l.Name, limit)
}

// Optimized Forward version
func (l *b_FullyConnectedLayer) Forward(input tensor.Tensor) tensor.Tensor {
    fmt.Printf("[BFCLayer:%s] >>> Forward start, input shape=%v\n", l.Name, input.Size())
    l.forwardInput = l.to.Clone(input)

    in := l.to.Reshape(input, []int{input.Size()[0], l.InputSize})
    defer l.to.Free(in)

    weightMat := l.to.Reshape(l.weights, []int{l.InputSize, l.OutputSize})
    defer l.to.Free(weightMat)

    var out tensor.Tensor
    if l.UseBias {
        fmt.Printf("[BFCLayer:%s] Forward GEMM with bias\n", l.Name)
        biasMat := l.to.Repeat(l.bias, input.Size()[0])
        defer l.to.Free(biasMat)
        
        biasMatReshape := l.to.Reshape(biasMat, []int{input.Size()[0], l.OutputSize})
        defer l.to.Free(biasMatReshape)
        
        out = l.to.Gemm(false, false, 1, 1, in, weightMat, biasMatReshape)
    } else {
        fmt.Printf("[BFCLayer:%s] Forward GEMM without bias\n", l.Name)
        zeroBias := l.to.Zeros([]int{input.Size()[0], l.OutputSize})
        defer l.to.Free(zeroBias)
        
        out = l.to.Gemm(false, false, 1, 1, in, weightMat, zeroBias)
    }

    fmt.Printf("[BFCLayer:%s] <<< Forward done, output shape=%v\n", l.Name, out.Size())
    return out
}

// Optimized Backward version
func (l *b_FullyConnectedLayer) Backward(input tensor.Tensor) tensor.Tensor {
	defer l.cleanupForwardCache() // Ensure cache is cleared

	fmt.Printf("[BFCLayer:%s] >>> Backward start, grad_in shape=%v\n", l.Name, input.Size())
	l.to.Clear(l.gradients)

	l.calculateWeightGradients(input)
	if l.UseBias {
		l.calculateBiasGradients(input)
	}

	output := l.calculateInputGradients(input)

	fmt.Printf("[BFCLayer:%s] <<< Backward done, grad_out shape=%v\n", l.Name, output.Size())
	return output
}

// Optimized calculateWeightGradients version
func (l *b_FullyConnectedLayer) calculateWeightGradients(input tensor.Tensor) {
	fmt.Printf("[BFCLayer:%s] -> Calculating weight gradients\n", l.Name)

	forwardInMatrix := l.to.Reshape(l.forwardInput, []int{l.forwardInput.Size()[0], l.InputSize})
	defer l.to.Free(forwardInMatrix)

	backwardInMatrix := l.to.Reshape(input, []int{input.Size()[0], l.OutputSize})
	defer l.to.Free(backwardInMatrix)

	zeroMatrix := l.to.Zeros([]int{l.InputSize, l.OutputSize})
	defer l.to.Free(zeroMatrix)

	g := l.to.Gemm(true, false, 1, 1, forwardInMatrix, backwardInMatrix, zeroMatrix)
	defer l.to.Free(g)

	l.to.Copy(l.weightGradients, g)
}

// Optimized calculateBiasGradients version
func (l *b_FullyConnectedLayer) calculateBiasGradients(input tensor.Tensor) {
    if !l.UseBias {
        return
    }
    fmt.Printf("[BFCLayer:%s] -> Calculating bias gradients\n", l.Name)
    g := l.to.Sum(input, []int{0})
    defer l.to.Free(g)
    
    l.to.Copy(l.biasGradients, g)
}

// Optimized calculateInputGradients version
func (l *b_FullyConnectedLayer) calculateInputGradients(input tensor.Tensor) tensor.Tensor {
    fmt.Printf("[BFCLayer:%s] -> Calculating input gradients\n", l.Name)
    
    weightMatrix := l.to.Reshape(l.weights, []int{l.InputSize, l.OutputSize})
    defer l.to.Free(weightMatrix)
    
    inputMatrix := l.to.Reshape(input, []int{input.Size()[0], l.OutputSize})
    defer l.to.Free(inputMatrix)
    
    zeroMatrix := l.to.Zeros([]int{input.Size()[0], l.InputSize})
    defer l.to.Free(zeroMatrix)

    out := l.to.Gemm(false, true, 1, 1, inputMatrix, weightMatrix, zeroMatrix)
    return out
}

// Optimized saveCalculateWeightGradients version
func (l *b_FullyConnectedLayer) saveCalculateWeightGradients(input tensor.Tensor) {
	forwardInMatrix := l.to.LazyReshape(l.forwardInput, []int{l.forwardInput.Size()[0], l.InputSize})
	defer l.to.Free(forwardInMatrix)

	backwardInMatrix := l.to.LazyReshape(input, []int{input.Size()[0], l.OutputSize})
	defer l.to.Free(backwardInMatrix)

	zeroMatrix := l.to.LazyZeros([]int{l.InputSize, l.OutputSize})
	defer l.to.Free(zeroMatrix)

	g := l.to.SaveGemm(true, false, 1, 1, forwardInMatrix, backwardInMatrix, zeroMatrix)
	defer l.to.Free(g)

	l.to.LazyCopy(l.weightGradients, g)
}

// Optimized lazyCalculateBiasGradients version
func (l *b_FullyConnectedLayer) lazyCalculateBiasGradients(input tensor.Tensor) {
	if !l.UseBias {
		return
	}
	g := l.to.LazySum(input, []int{0})
	defer l.to.Free(g)

	l.to.LazyCopy(l.biasGradients, g)
}

// Optimized saveCalculateInputGradients version
func (l *b_FullyConnectedLayer) saveCalculateInputGradients(input tensor.Tensor) tensor.Tensor {
	weightMatrix := l.to.LazyReshape(l.weights, []int{l.InputSize, l.OutputSize})
	defer l.to.Free(weightMatrix)

	inputMatrix := l.to.LazyReshape(input, []int{input.Size()[0], l.OutputSize})
	defer l.to.Free(inputMatrix)

	zeroMatrix := l.to.LazyZeros([]int{input.Size()[0], l.InputSize})
	defer l.to.Free(zeroMatrix)

	out := l.to.SaveGemm(false, true, 1, 1, inputMatrix, weightMatrix, zeroMatrix)
	return out
}

// cleanupForwardCache Clear forward propagation cache
func (l *b_FullyConnectedLayer) cleanupForwardCache() {
	if l.forwardInput != nil {
		l.to.Free(l.forwardInput)
		l.forwardInput = nil
	}
}

// Close Release all resources
func (l *b_FullyConnectedLayer) Close() {
	l.cleanupForwardCache()

	// Release parameters and gradients
	if l.parameters != nil {
		l.to.Free(l.parameters)
	}
	if l.gradients != nil {
		l.to.Free(l.gradients)
	}
}

func (l b_FullyConnectedLayer) Parameters() tensor.Tensor {
	return l.parameters
}

func (l b_FullyConnectedLayer) Gradients() tensor.Tensor {
	return l.gradients
}
