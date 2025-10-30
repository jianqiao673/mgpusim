package layers

import (
	"fmt"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// An EmbeddingLayer implements an embedding layer.
type EmbeddingLayer struct {
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

// NewEmbeddingLayer creates a new embedding layer.
func NewEmbeddingLayer(
	index int,
	to tensor.Operator,
	vocabSize, embeddingDim int,
) *EmbeddingLayer {
	numWeight := vocabSize * embeddingDim

	l := &EmbeddingLayer{
		layerIndex:   index,
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
	fmt.Printf("[NewEmbeddingLayer-Allocate] gradients: 0x%x, weightGradients: 0x%x\n",
		l.gradients, l.weightGradients)

	return l
}

// Randomize initializes the parameters of the layer randomly.
func (l *EmbeddingLayer) Randomize() {
	numWeight := l.VocabSize * l.EmbeddingDim
	weights := make([]float64, numWeight)
	for i := 0; i < numWeight; i++ {
		weights[i] = (rand.Float64() - 0.5) / float64(l.EmbeddingDim) * 2
	}
	l.to.Init(l.weights, weights)
}

// Forward performs the forward propagation operation.
func (l *EmbeddingLayer) Forward(input tensor.Tensor) tensor.Tensor {
    l.forwardInput = l.to.Clone(input)
    
    // 直接调用，返回输出张量
    output := l.to.EmbeddingForward(input, l.weights, -1)
    
    l.to.Free(input)
    return output
}

// Backward calculates the weight gradients.
func (l *EmbeddingLayer) Backward(
	input tensor.Tensor,
) tensor.Tensor {
	l.to.Clear(l.gradients)

	l.calculateWeightGradients(input)
	var output tensor.Tensor

	if l.layerIndex > 0 {
		output = l.calculateInputGradients(input)
	}

	l.to.Free(l.forwardInput)
	l.to.Free(input) // Free input

	return output
}

func (l *EmbeddingLayer) calculateWeightGradients(input tensor.Tensor) {
    // 计算权重梯度并返回梯度张量
    gradWeight := l.to.EmbeddingBackwardWeight(
        l.forwardInput, input, -1, 1.0,
    )
    
    // 将梯度复制到权重梯度张量中
    l.to.Copy(l.weightGradients, gradWeight)
    l.to.Free(gradWeight)
}

func (l *EmbeddingLayer) calculateInputGradients(
	input tensor.Tensor,
) tensor.Tensor {
	batchSize := l.forwardInput.Size()[0]
	seqLen := l.forwardInput.Size()[1]

	// embedding 层对输入的梯度通常为0，因为输入是索引
	output := l.to.Zeros([]int{batchSize, seqLen})
	return output
}

// Parameters returns the parameters of the layer.
func (l EmbeddingLayer) Parameters() tensor.Tensor {
	return l.parameters
}

// Gradients returns the gradients of the layer.
func (l EmbeddingLayer) Gradients() tensor.Tensor {
	return l.gradients
}

// SaveNewEmbeddingLayer creates an embedding layer with memory optimization.
func SaveNewEmbeddingLayer(
	index int,
	to tensor.Operator,
	vocabSize, embeddingDim int,
) *EmbeddingLayer {
	numWeight := vocabSize * embeddingDim

	l := &EmbeddingLayer{
		layerIndex:   index,
		to:           to,
		VocabSize:    vocabSize,
		EmbeddingDim: embeddingDim,
		gradients:    to.Create([]int{numWeight}),
	}

	l.weightGradients = l.gradients

	fmt.Printf("[SaveNewEmbeddingLayer-Allocate] gradients: 0x%x, weightGradients: 0x%x\n",
		l.gradients, l.weightGradients)

	return l
}

// LazyRandomize lazily initializes the parameters of the layer randomly.
func (l *EmbeddingLayer) LazyRandomize() {
	fmt.Printf("EmbeddingLayer.LazyRandomize\n")

	numWeight := l.VocabSize * l.EmbeddingDim
	weights := make([]float64, numWeight)
	for i := 0; i < numWeight; i++ {
		weights[i] = (rand.Float64() - 0.5) / float64(l.EmbeddingDim) * 2
	}

	
	l.parameters = l.to.CreateWithData(weights, []int{numWeight}, "")
	l.weights = l.parameters

	fmt.Printf("[LazyRandomize-Allocate] parameters: 0x%x, weights: 0x%x\n",
		l.parameters, l.weights)
}

// SaveForward performs the forward propagation operation in a memory saving way.
func (l *EmbeddingLayer) SaveForward(
	input tensor.Tensor,
) tensor.Tensor {
	l.forwardInput = l.to.LazyClone(input)

	// 执行 embedding 查找（内存优化版本）
	output := l.to.EmbeddingForward(input, l.weights, -1)

	l.to.Free(input) // Free input

	return output
}

// SaveBackward calculates the weight gradients in a memory saving way.
func (l *EmbeddingLayer) SaveBackward(
	input tensor.Tensor,
) tensor.Tensor {
	l.to.Clear(l.gradients) // Original clear does not allocate new memory

	l.saveCalculateWeightGradients(input)
	var output tensor.Tensor

	if l.layerIndex > 0 {
		output = l.saveCalculateInputGradients(input)
	}

	l.to.Free(l.forwardInput)
	l.to.Free(input) // Free input

	return output
}

func (l *EmbeddingLayer) saveCalculateWeightGradients(
	input tensor.Tensor,
) {
	// 计算权重梯度并返回梯度张量
	gradWeight := l.to.EmbeddingBackwardWeight(
		l.forwardInput, // 前向传播的输入索引
		input,          // 梯度输入
		-1,             // padding index
		1.0,            // scale
	)

	// 将计算得到的梯度复制到权重梯度张量中
	l.to.LazyCopy(l.weightGradients, gradWeight)
	l.to.Free(gradWeight) // 释放临时梯度张量
}

func (l *EmbeddingLayer) saveCalculateInputGradients(
	input tensor.Tensor,
) tensor.Tensor {
	batchSize := l.forwardInput.Size()[0]
	seqLen := l.forwardInput.Size()[1]

	// embedding 层对输入的梯度通常为0
	output := l.to.LazyZeros([]int{batchSize, seqLen})
	return output
}

// SetWeights allows setting pre-trained weights for the embedding layer
func (l *EmbeddingLayer) SetWeights(weights []float64) {
	l.to.Init(l.weights, weights)
}

// GetWeights returns the current weights of the embedding layer
func (l *EmbeddingLayer) GetWeights() tensor.Tensor {
	return l.weights
}

// GetOutputShape returns the output shape for a given input shape
func (l *EmbeddingLayer) GetOutputShape(inputShape []int) []int {
	// input: [batch_size, seq_len]
	// output: [batch_size, seq_len, embedding_dim]
	return []int{inputShape[0], inputShape[1], l.EmbeddingDim}
}