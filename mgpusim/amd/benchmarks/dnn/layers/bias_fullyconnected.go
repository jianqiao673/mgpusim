package layers

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// FullyConnectedLayer implements a fully connected layer compatible with MiniGPT.
type b_FullyConnectedLayer struct {
	Name string
	to   tensor.Operator

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

	// 参数张量
	l.parameters = to.Create([]int{numParams})
	l.weights = to.Slice(l.parameters, 0, numWeight)
	if useBias {
		l.bias = to.Slice(l.parameters, numWeight, numParams)
	}

	// 梯度张量
	l.gradients = to.Create([]int{numParams})
	l.weightGradients = to.Slice(l.gradients, 0, numWeight)
	if useBias {
		l.biasGradients = to.Slice(l.gradients, numWeight, numParams)
	}

	// 初始化为 0
	/*fmt.Printf("[BFCLayer:%s] Step: before Init weights (num=%d)\n", l.Name, numWeight)
	to.Init(l.weights, make([]float64, numWeight))
	fmt.Printf("[BFCLayer:%s] Step: after Init weights\n", l.Name)

	fmt.Printf("[BFCLayer:%s] Step: before Init weightGradients (num=%d)\n", l.Name, numWeight)
	to.Init(l.weightGradients, make([]float64, numWeight))
	fmt.Printf("[BFCLayer:%s] Step: after Init weightGradients\n", l.Name)

	if useBias {
		fmt.Printf("[BFCLayer:%s] Step: before Init bias (num=%d)\n", l.Name, numBias)
		to.Init(l.bias, make([]float64, numBias))
		fmt.Printf("[BFCLayer:%s] Step: after Init bias\n", l.Name)

		fmt.Printf("[BFCLayer:%s] Step: before Init biasGradients (num=%d)\n", l.Name, numBias)
		to.Init(l.biasGradients, make([]float64, numBias))
		fmt.Printf("[BFCLayer:%s] Step: after Init biasGradients\n", l.Name)
	}*/
	fmt.Printf("[BFCLayer:%s] ✅ All inits done\n", l.Name)

	fmt.Printf("[BFCLayer:%s] Allocated weights=%p, bias=%p, grads=(%p,%p)\n",
		l.Name, l.weights, l.bias, l.weightGradients, l.biasGradients)

	return l
}

func (l *b_FullyConnectedLayer) LazyRandomize() {
}

func (l *b_FullyConnectedLayer) SaveBackward(t tensor.Tensor) tensor.Tensor {
	return t
}

func (l *b_FullyConnectedLayer) SaveForward(t tensor.Tensor) tensor.Tensor {
	return t
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

// Forward performs the forward propagation
func (l *b_FullyConnectedLayer) Forward(input tensor.Tensor) tensor.Tensor {
	fmt.Printf("[BFCLayer:%s] >>> Forward start, input shape=%v\n", l.Name, input.Size())
	l.forwardInput = l.to.Clone(input)

	in := l.to.Reshape(input, []int{input.Size()[0], l.InputSize})
	weightMat := l.to.Reshape(l.weights, []int{l.InputSize, l.OutputSize})

	var out tensor.Tensor
	if l.UseBias {
		fmt.Printf("[BFCLayer:%s] Forward GEMM with bias\n", l.Name)
		biasMat := l.to.Repeat(l.bias, input.Size()[0])
		biasMatReshape := l.to.Reshape(biasMat, []int{input.Size()[0], l.OutputSize})
		out = l.to.Gemm(false, false, 1, 1, in, weightMat, biasMatReshape)
		//l.to.Free(biasMat)
		//l.to.Free(biasMatReshape)
	} else {
		fmt.Printf("[BFCLayer:%s] Forward GEMM without bias\n", l.Name)
		zeroBias := l.to.Zeros([]int{input.Size()[0], l.OutputSize})
		out = l.to.Gemm(false, false, 1, 1, in, weightMat, zeroBias)
		//l.to.Free(zeroBias)
	}

	//l.to.Free(in)
	//l.to.Free(weightMat)
	//l.to.Free(input)

	fmt.Printf("[BFCLayer:%s] <<< Forward done, output shape=%v\n", l.Name, out.Size())
	return out
}

// Backward performs the backward propagation
func (l *b_FullyConnectedLayer) Backward(input tensor.Tensor) tensor.Tensor {
	fmt.Printf("[BFCLayer:%s] >>> Backward start, grad_in shape=%v\n", l.Name, input.Size())
	l.to.Clear(l.gradients)

	l.calculateWeightGradients(input)
	if l.UseBias {
		l.calculateBiasGradients(input)
	}

	output := l.calculateInputGradients(input)

	//l.to.Free(l.forwardInput)
	//l.to.Free(input)

	fmt.Printf("[BFCLayer:%s] <<< Backward done, grad_out shape=%v\n", l.Name, output.Size())
	return output
}

func (l *b_FullyConnectedLayer) calculateWeightGradients(input tensor.Tensor) {
	fmt.Printf("[BFCLayer:%s] -> Calculating weight gradients\n", l.Name)
	forwardInMatrix := l.to.Reshape(l.forwardInput, []int{l.forwardInput.Size()[0], l.InputSize})
	backwardInMatrix := l.to.Reshape(input, []int{input.Size()[0], l.OutputSize})
	zeroMatrix := l.to.Zeros([]int{l.InputSize, l.OutputSize})

	g := l.to.Gemm(true, false, 1, 1, forwardInMatrix, backwardInMatrix, zeroMatrix)
	l.to.Copy(l.weightGradients, g)

	//l.to.Free(forwardInMatrix)
	//l.to.Free(backwardInMatrix)
	//l.to.Free(zeroMatrix)
	//l.to.Free(g)
}

func (l *b_FullyConnectedLayer) calculateBiasGradients(input tensor.Tensor) {
	if !l.UseBias {
		return
	}
	fmt.Printf("[BFCLayer:%s] -> Calculating bias gradients\n", l.Name)
	g := l.to.Sum(input, []int{0})
	l.to.Copy(l.biasGradients, g)
	//l.to.Free(g)
}

func (l *b_FullyConnectedLayer) calculateInputGradients(input tensor.Tensor) tensor.Tensor {
	fmt.Printf("[BFCLayer:%s] -> Calculating input gradients\n", l.Name)
	weightMatrix := l.to.Reshape(l.weights, []int{l.InputSize, l.OutputSize})
	inputMatrix := l.to.Reshape(input, []int{input.Size()[0], l.OutputSize})
	zeroMatrix := l.to.Zeros([]int{input.Size()[0], l.InputSize})

	out := l.to.Gemm(false, true, 1, 1, inputMatrix, weightMatrix, zeroMatrix)

	//l.to.Free(weightMatrix)
	//l.to.Free(inputMatrix)
	//l.to.Free(zeroMatrix)
	return out
}

func (l b_FullyConnectedLayer) Parameters() tensor.Tensor {
	return l.parameters
}

func (l b_FullyConnectedLayer) Gradients() tensor.Tensor {
	return l.gradients
}
