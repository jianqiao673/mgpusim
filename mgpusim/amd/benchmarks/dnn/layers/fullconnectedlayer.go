package layers

import (
	"fmt"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// A FullyConnectedLayer implements a fully connected layer.
type FullyConnectedLayer struct {
	layerIndex int
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
}

// NewFullyConnectedLayer creates a fully connected layer.
func NewFullyConnectedLayer(
	index int,
	to tensor.Operator,
	inputSize, outputSize int,
) *FullyConnectedLayer {
	numWeight := inputSize * outputSize
	numBias := outputSize
	numParams := numWeight + numBias

	l := &FullyConnectedLayer{
		layerIndex: index,
		to:         to,
		InputSize:  inputSize,
		OutputSize: outputSize,
		parameters: to.Create([]int{numParams}),
		gradients:  to.Create([]int{numParams}),
	}

	l.weights = to.Slice(l.parameters, 0, numWeight)
	l.bias = to.Slice(l.parameters, numWeight, numParams)
	l.weightGradients = to.Slice(l.gradients, 0, numWeight)
	l.biasGradients = to.Slice(l.gradients, numWeight, numParams)

	fmt.Printf("[NewFullyConnectedLayer] parameters: 0x%x, weights: 0x%x, bias: 0x%x\n", 
		l.parameters, l.weights, l.bias)
		
	fmt.Printf("[NewFullyConnectedLayer] gradients: 0x%x, weightGradients: 0x%x, biasGradients: 0x%x\n", 
		l.gradients, l.weightGradients, l.biasGradients)

	return l
}

// Randomize initialize the parameters of the layer randomly.
func (l *FullyConnectedLayer) Randomize() {
	numWeight := l.InputSize * l.OutputSize
	weights := make([]float64, numWeight)
	for i := 0; i < numWeight; i++ {
		weights[i] = (rand.Float64() - 0.5) / float64(l.InputSize) * 2
	}
	l.to.Init(l.weights, weights)

	numBias := l.OutputSize
	bias := make([]float64, numBias)
	for i := 0; i < numBias; i++ {
		bias[i] = rand.Float64()*2 - 1
	}
	l.to.Init(l.bias, bias)
}

// Forward performs the forward propagation operation.
func (l *FullyConnectedLayer) Forward(
	input tensor.Tensor,
) tensor.Tensor {
	l.forwardInput = l.to.Clone(input)

	in := l.to.Reshape(input, []int{input.Size()[0], l.InputSize})
	weightMat := l.to.Reshape(l.weights, []int{l.InputSize, l.OutputSize})
	biasMat := l.to.Repeat(l.bias, input.Size()[0])
	biasMatReshape := l.to.Reshape(biasMat,
		[]int{input.Size()[0], l.OutputSize})

	out := l.to.Gemm(false, false, 1, 1, in, weightMat, biasMatReshape)

	l.to.Free(in)
	l.to.Free(weightMat)

	// fmt.Printf("biasMat: 0x%x, biasMatReshape: 0x%x\n", 
	// 	biasMat, biasMatReshape)
	l.to.Free(biasMat)
	l.to.Free(biasMatReshape)

	return out
}

// Backward calculate the weight, bias, and input gradients.
func (l *FullyConnectedLayer) Backward(
	input tensor.Tensor,
) tensor.Tensor {
	l.to.Clear(l.gradients)

	l.calculateWeightGradients(input)
	l.calculateBiasGradients(input)
	var output tensor.Tensor

	if l.layerIndex > 0 {
		output = l.calculateInputGradients(input)
	}

	l.to.Free(l.forwardInput)

	return output
}

func (l *FullyConnectedLayer) calculateWeightGradients(
	input tensor.Tensor,
) {
	forwardInMatrix := l.to.Reshape(l.forwardInput,
		[]int{l.forwardInput.Size()[0], l.InputSize})
	backwardInMatrix := l.to.Reshape(input,
		[]int{input.Size()[0], l.OutputSize})
	zeroMatrix := l.to.Zeros([]int{l.InputSize, l.OutputSize})

	g := l.to.Gemm(
		true, false,
		1, 1,
		forwardInMatrix, backwardInMatrix,
		zeroMatrix,
	)

	l.to.Copy(l.weightGradients, g)

	l.to.Free(forwardInMatrix)
	l.to.Free(backwardInMatrix)
	l.to.Free(zeroMatrix)
	l.to.Free(g)
}

func (l *FullyConnectedLayer) calculateBiasGradients(
	input tensor.Tensor,
) {
	g := l.to.Sum(input, []int{0})
	l.to.Copy(l.biasGradients, g)
	l.to.Free(g)
}

func (l *FullyConnectedLayer) calculateInputGradients(
	input tensor.Tensor,
) tensor.Tensor {
	weightMatrix := l.to.Reshape(l.weights, []int{l.InputSize, l.OutputSize})
	inputMatrix := l.to.Reshape(input, []int{input.Size()[0], l.OutputSize})
	zeroMatrix := l.to.Zeros([]int{input.Size()[0], l.InputSize})

	out := l.to.Gemm(false, true, 1, 1, inputMatrix, weightMatrix, zeroMatrix)

	l.to.Free(weightMatrix)
	l.to.Free(inputMatrix)
	l.to.Free(zeroMatrix)

	return out
}

// Parameters returns the parameters of the layer.
func (l FullyConnectedLayer) Parameters() tensor.Tensor {
	return l.parameters
}

// Gradients returns the gradients of the layer.
func (l FullyConnectedLayer) Gradients() tensor.Tensor {
	return l.gradients
}

// NewFullyConnectedLayer creates a fully connected layer.
func SaveNewFullyConnectedLayer(
	index int,
	to tensor.Operator,
	inputSize, outputSize int,
) *FullyConnectedLayer {
	numWeight := inputSize * outputSize
	numBias := outputSize
	numParams := numWeight + numBias

	l := &FullyConnectedLayer{
		layerIndex: index,
		to:         to,
		InputSize:  inputSize,
		OutputSize: outputSize,
		// parameters: to.Create([]int{numParams}),
		gradients:  to.Create([]int{numParams}),
	}

	l.weightGradients = to.Slice(l.gradients, 0, numWeight)
	l.biasGradients = to.Slice(l.gradients, numWeight, numParams)

	return l
}

// LazyRandomize lazily initialize the parameters of the layer randomly.
func (l *FullyConnectedLayer) LazyRandomize() {
	fmt.Printf("FullyConnectedLayer.LazyRandomize\n")
	
	numWeight := l.InputSize * l.OutputSize
	
	weights := make([]float64, numWeight)
	for i := 0; i < numWeight; i++ {
		weights[i] = (rand.Float64() - 0.5) / float64(l.InputSize) * 2
	}
	
	numBias := l.OutputSize
	bias := make([]float64, numBias)
	for i := 0; i < numBias; i++ {
		bias[i] = rand.Float64()*2 - 1
	}

	numParams := numWeight + numBias
	datas := [][]float64{weights, bias}
	nums := []int{numWeight, numParams}
	
	slices := l.to.LazyInitSlices(datas, nums, numParams)
	l.parameters = slices[0]
	l.weights = slices[1]
	l.bias = slices[2]

	fmt.Printf("[LazyRandomize] parameters: 0x%x, weights: 0x%x, bias: 0x%x\n", l.parameters, l.weights, l.bias)
}

// SaveForward performs the forward propagation operation
// in a memory saving way.
func (l *FullyConnectedLayer) SaveForward(
	input tensor.Tensor,
) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)

	in := l.to.LazyReshape(input, []int{input.Size()[0], l.InputSize})
	weightMat := l.to.LazyReshape(l.weights, []int{l.InputSize, l.OutputSize})
	biasMat := l.to.LazyRepeat(l.bias, input.Size()[0])
	biasMatReshape := l.to.LazyReshape(biasMat,
		[]int{input.Size()[0], l.OutputSize})

	out := l.to.SaveGemm(false, false, 1, 1, in, weightMat, biasMatReshape)

	l.to.Free(in)
	l.to.Free(weightMat)
	l.to.Free(input) // Free input.
	l.to.Free(biasMat) // Free biasMat
	l.to.Free(biasMatReshape) // Free biasMatReshape

	// fmt.Printf("biasMat: 0x%x, biasMatReshape: 0x%x\n", 
	// 	biasMat, biasMatReshape)
	// l.to.Free(biasMat) // if free, then page not found
	// l.to.Free(biasMatReshape) // if free, then value mismatch

	return out
}

// Backward calculate the weight, bias, and input gradients.
func (l *FullyConnectedLayer) SaveBackward(
	input tensor.Tensor,
) tensor.Tensor {
	l.to.Clear(l.gradients) // Original clear does not allocate new memory.

	l.saveCalculateWeightGradients(input)
	l.lazyCalculateBiasGradients(input)
	var output tensor.Tensor

	if l.layerIndex > 0 {
		output = l.saveCalculateInputGradients(input)
	}

	l.to.Free(l.forwardInput)
	l.to.Free(input) // Free input.

	return output
}

func (l *FullyConnectedLayer) saveCalculateWeightGradients(
	input tensor.Tensor,
) {
	forwardInMatrix := l.to.LazyReshape(l.forwardInput,
		[]int{l.forwardInput.Size()[0], l.InputSize})
	backwardInMatrix := l.to.LazyReshape(input,
		[]int{input.Size()[0], l.OutputSize})
	zeroMatrix := l.to.LazyZeros([]int{l.InputSize, l.OutputSize})

	g := l.to.SaveGemm(
		true, false,
		1, 1,
		forwardInMatrix, backwardInMatrix,
		zeroMatrix,
	)

	l.to.LazyCopy(l.weightGradients, g)

	l.to.Free(forwardInMatrix)
	l.to.Free(backwardInMatrix)
	l.to.Free(zeroMatrix)
	l.to.Free(g)
}

func (l *FullyConnectedLayer) lazyCalculateBiasGradients(
	input tensor.Tensor,
) {
	g := l.to.LazySum(input, []int{0})
	l.to.LazyCopy(l.biasGradients, g)
	l.to.Free(g)
}

func (l *FullyConnectedLayer) saveCalculateInputGradients(
	input tensor.Tensor,
) tensor.Tensor {
	weightMatrix := l.to.LazyReshape(l.weights, []int{l.InputSize, l.OutputSize})
	inputMatrix := l.to.LazyReshape(input, []int{input.Size()[0], l.OutputSize})
	zeroMatrix := l.to.LazyZeros([]int{input.Size()[0], l.InputSize})

	out := l.to.SaveGemm(false, true, 1, 1, inputMatrix, weightMatrix, zeroMatrix)

	l.to.Free(weightMatrix)
	l.to.Free(inputMatrix)
	l.to.Free(zeroMatrix)

	return out
}