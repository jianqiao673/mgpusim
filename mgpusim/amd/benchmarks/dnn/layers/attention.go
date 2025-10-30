package layers

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// CausalSelfAttentionConfig 配置结构体
type CausalSelfAttentionConfig struct {
	NEmbd     int  // 嵌入维度
	NHead     int  // 注意力头数
	Bias      bool // 是否使用偏置
	BlockSize int  // 序列长度
}

// CausalSelfAttentionLayer 因果自注意力层
type CausalSelfAttentionLayer struct {
	layerIndex int
	to         tensor.Operator
	config     CausalSelfAttentionConfig

	// 权重参数
	cAttnWeights tensor.Tensor // QKV 投影权重 [n_embd, 3*n_embd]
	cAttnBias    tensor.Tensor // QKV 投影偏置 [3*n_embd]
	cProjWeights tensor.Tensor // 输出投影权重 [n_embd, n_embd]
	cProjBias    tensor.Tensor // 输出投影偏置 [n_embd]

	// 梯度
	cAttnWeightsGrad tensor.Tensor
	cAttnBiasGrad    tensor.Tensor
	cProjWeightsGrad tensor.Tensor
	cProjBiasGrad    tensor.Tensor

	// 缓存用于反向传播
	forwardInput     tensor.Tensor
	attentionWeights tensor.Tensor
	softmaxOutput    tensor.Tensor
}

// NewCausalSelfAttentionLayer 创建新的因果自注意力层
func NewCausalSelfAttentionLayer(
	index int,
	to tensor.Operator,
	config CausalSelfAttentionConfig,
) *CausalSelfAttentionLayer {
	// 检查嵌入维度是否能被头数整除
	if config.NEmbd%config.NHead != 0 {
		panic(fmt.Sprintf("n_embd (%d) must be divisible by n_head (%d)",
			config.NEmbd, config.NHead))
	}

	l := &CausalSelfAttentionLayer{
		layerIndex: index,
		to:         to,
		config:     config,
	}

	// 初始化 QKV 投影层参数
	l.cAttnWeights = to.Create([]int{config.NEmbd, 3 * config.NEmbd})
	l.cAttnWeightsGrad = to.Create([]int{config.NEmbd, 3 * config.NEmbd})

	if config.Bias {
		l.cAttnBias = to.Create([]int{3 * config.NEmbd})
		l.cAttnBiasGrad = to.Create([]int{3 * config.NEmbd})
	}

	// 初始化输出投影层参数
	l.cProjWeights = to.Create([]int{config.NEmbd, config.NEmbd})
	l.cProjWeightsGrad = to.Create([]int{config.NEmbd, config.NEmbd})

	if config.Bias {
		l.cProjBias = to.Create([]int{config.NEmbd})
		l.cProjBiasGrad = to.Create([]int{config.NEmbd})
	}

	fmt.Printf("[NewCausalSelfAttentionLayer] layer %d created with %d heads, %d embed dim\n",
		index, config.NHead, config.NEmbd)

	return l
}

// Randomize 随机初始化参数
func (l *CausalSelfAttentionLayer) Randomize() {
	// 使用 Xavier 初始化
	xavier := math.Sqrt(2.0 / float64(l.config.NEmbd))

	// 初始化 QKV 投影权重
	cAttnData := make([]float64, l.config.NEmbd*3*l.config.NEmbd)
	for i := range cAttnData {
		cAttnData[i] = (rand.Float64() - 0.5) * 2 * xavier
	}
	l.to.Init(l.cAttnWeights, cAttnData)

	// 初始化输出投影权重
	cProjData := make([]float64, l.config.NEmbd*l.config.NEmbd)
	for i := range cProjData {
		cProjData[i] = (rand.Float64() - 0.5) * 2 * xavier
	}
	l.to.Init(l.cProjWeights, cProjData)

	// 初始化偏置（如果使用）
	if l.config.Bias {
		l.to.Init(l.cAttnBias, make([]float64, 3*l.config.NEmbd))
		l.to.Init(l.cProjBias, make([]float64, l.config.NEmbd))
	}
}

// Forward 前向传播
func (l *CausalSelfAttentionLayer) Forward(input tensor.Tensor) tensor.Tensor {
	l.forwardInput = l.to.Clone(input)

	B, T, C := input.Size()[0], input.Size()[1], input.Size()[2]
	headSize := C / l.config.NHead

	// 1. QKV 投影 [B, T, C] -> [B, T, 3*C]
	qkv := l.computeQKV(input)
	defer l.to.Free(qkv)

	// 2. 分割 Q, K, V
	q, k, v := l.splitQKV(qkv, B, T, C)
	defer l.to.Free(q)
	defer l.to.Free(k)
	defer l.to.Free(v)

	// 3. 重塑为多头格式 [B, T, C] -> [B, T, n_head, head_size] -> [B, n_head, T, head_size]
	q = l.reshapeToMultiHead(q, B, T, headSize)
	k = l.reshapeToMultiHead(k, B, T, headSize)
	v = l.reshapeToMultiHead(v, B, T, headSize)
	defer l.to.Free(q)
	defer l.to.Free(k)
	defer l.to.Free(v)

	// 4. 计算注意力分数: Q @ K^T / sqrt(head_size)
	att := l.computeAttentionScores(q, k, B, T, headSize)
	defer l.to.Free(att)

	// 5. 应用因果掩码
	att = l.applyCausalMask(att, T)
	defer l.to.Free(att)

	// 6. Softmax
	l.attentionWeights = l.to.Softmax(att)
	l.softmaxOutput = l.to.Clone(l.attentionWeights)

	// 7. 注意力加权: att @ v
	y := l.applyAttention(l.attentionWeights, v, B, T, headSize)
	defer l.to.Free(y)

	// 8. 重新组装多头输出 [B, n_head, T, head_size] -> [B, T, n_head, head_size] -> [B, T, C]
	y = l.reassembleHeads(y, B, T, C)

	// 9. 输出投影
	output := l.outputProjection(y)

	l.to.Free(input)
	return output
}

// computeQKV 计算 QKV 投影
func (l *CausalSelfAttentionLayer) computeQKV(input tensor.Tensor) tensor.Tensor {
	B, T, C := input.Size()[0], input.Size()[1], input.Size()[2]

	// 重塑输入为 2D 以便进行矩阵乘法
	input2D := l.to.Reshape(input, []int{B * T, C})
	defer l.to.Free(input2D)

	// 计算 QKV 投影
	qkv2D := l.to.Gemm(false, false, 1.0, 0.0, input2D, l.cAttnWeights, l.cAttnBias)
	qkv := l.to.Reshape(qkv2D, []int{B, T, 3 * C})
	l.to.Free(qkv2D)

	return qkv
}

// splitQKV 分割 QKV
func (l *CausalSelfAttentionLayer) splitQKV(qkv tensor.Tensor, B, T, C int) (q, k, v tensor.Tensor) {
	// 手动分割 QKV
	qkvData := qkv.Vector()

	qData := make([]float64, B*T*C)
	kData := make([]float64, B*T*C)
	vData := make([]float64, B*T*C)

	for i := 0; i < B*T; i++ {
		copy(qData[i*C:(i+1)*C], qkvData[i*3*C:i*3*C+C])
		copy(kData[i*C:(i+1)*C], qkvData[i*3*C+C:i*3*C+2*C])
		copy(vData[i*C:(i+1)*C], qkvData[i*3*C+2*C:(i+1)*3*C])
	}

	q = l.to.CreateWithData(qData, []int{B, T, C}, "q")
	k = l.to.CreateWithData(kData, []int{B, T, C}, "k")
	v = l.to.CreateWithData(vData, []int{B, T, C}, "v")

	return q, k, v
}

// reshapeToMultiHead 重塑为多头格式
func (l *CausalSelfAttentionLayer) reshapeToMultiHead(t tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// [B, T, C] -> [B, T, n_head, head_size] -> [B, n_head, T, head_size]
	multiHead := l.to.Reshape(t, []int{B, T, l.config.NHead, headSize})
	transposed := l.to.Transpose(multiHead, []int{0, 2, 1, 3})
	l.to.Free(multiHead)
	return transposed
}

// computeAttentionScores 计算注意力分数
func (l *CausalSelfAttentionLayer) computeAttentionScores(q, k tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// 转置 K: [B, n_head, T, head_size] -> [B, n_head, head_size, T]
	kT := l.to.Transpose(k, []int{0, 1, 3, 2})
	defer l.to.Free(kT)

	// 计算 Q @ K^T: [B, n_head, T, head_size] @ [B, n_head, head_size, T] -> [B, n_head, T, T]
	att := l.batchMatMul4D(q, kT)

	// 缩放: / sqrt(head_size)
	scale := 1.0 / math.Sqrt(float64(headSize))
	scaledAtt := l.scaleTensor(att, scale)
	l.to.Free(att)

	return scaledAtt
}

// batchMatMul4D 4D 批量矩阵乘法
func (l *CausalSelfAttentionLayer) batchMatMul4D(a, b tensor.Tensor) tensor.Tensor {
	aSize := a.Size()

	B, nHead, T, headSize := aSize[0], aSize[1], aSize[2], aSize[3]

	// 重塑为 2D 以便使用 Gemm
	a2D := l.to.Reshape(a, []int{B * nHead * T, headSize})
	b2D := l.to.Reshape(b, []int{B * nHead * headSize, T})

	// 计算矩阵乘法
	result2D := l.to.Gemm(false, false, 1.0, 0.0, a2D, b2D, l.to.Zeros([]int{B * nHead * T, T}))

	// 重塑回 4D
	result := l.to.Reshape(result2D, []int{B, nHead, T, T})

	l.to.Free(a2D)
	l.to.Free(b2D)
	l.to.Free(result2D)

	return result
}

// scaleTensor 缩放张量
func (l *CausalSelfAttentionLayer) scaleTensor(t tensor.Tensor, scale float64) tensor.Tensor {
	data := t.Vector()
	scaledData := make([]float64, len(data))

	for i := range data {
		scaledData[i] = data[i] * scale
	}

	return l.to.CreateWithData(scaledData, t.Size(), "scaled")
}

// applyCausalMask 应用因果掩码
func (l *CausalSelfAttentionLayer) applyCausalMask(att tensor.Tensor, T int) tensor.Tensor {
	mask := l.createCausalMask(T)
	defer l.to.Free(mask)

	return l.applyCausalMaskToAttention(att, mask, T)
}

// createCausalMask 创建因果掩码
func (l *CausalSelfAttentionLayer) createCausalMask(T int) tensor.Tensor {
	// 创建下三角矩阵 [1, 1, T, T]
	maskData := make([]float64, T*T)
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			if j > i { // 上三角部分为 -inf，下三角部分为 0
				maskData[i*T+j] = -1e9
			} else {
				maskData[i*T+j] = 0
			}
		}
	}

	return l.to.CreateWithData(maskData, []int{1, 1, T, T}, "causal_mask")
}

// applyCausalMaskToAttention 应用因果掩码到注意力分数
func (l *CausalSelfAttentionLayer) applyCausalMaskToAttention(att, mask tensor.Tensor, T int) tensor.Tensor {
	attSize := att.Size()
	B, nHead := attSize[0], attSize[1]

	// 广播掩码到与注意力分数相同的形状
	broadcastMask := l.broadcastMask(mask, B, nHead, T)
	defer l.to.Free(broadcastMask)

	// 应用掩码: att + mask
	maskedAtt := l.addTensors(att, broadcastMask)

	return maskedAtt
}

// broadcastMask 广播掩码
func (l *CausalSelfAttentionLayer) broadcastMask(mask tensor.Tensor, B, nHead, T int) tensor.Tensor {
	maskData := mask.Vector()
	broadcastData := make([]float64, B*nHead*T*T)

	for b := 0; b < B; b++ {
		for h := 0; h < nHead; h++ {
			for i := 0; i < T; i++ {
				for j := 0; j < T; j++ {
					idx := ((b*nHead+h)*T+i)*T + j
					broadcastData[idx] = maskData[i*T+j]
				}
			}
		}
	}

	return l.to.CreateWithData(broadcastData, []int{B, nHead, T, T}, "broadcast_mask")
}

// addTensors 张量加法
func (l *CausalSelfAttentionLayer) addTensors(a, b tensor.Tensor) tensor.Tensor {
	aData := a.Vector()
	bData := b.Vector()
	resultData := make([]float64, len(aData))

	for i := range aData {
		resultData[i] = aData[i] + bData[i]
	}

	return l.to.CreateWithData(resultData, a.Size(), "added")
}

// applyAttention 应用注意力权重
func (l *CausalSelfAttentionLayer) applyAttention(att, v tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// [B, n_head, T, T] @ [B, n_head, T, head_size] -> [B, n_head, T, head_size]
	return l.batchMatMul4D(att, v)
}

// reassembleHeads 重新组装多头输出
func (l *CausalSelfAttentionLayer) reassembleHeads(y tensor.Tensor, B, T, C int) tensor.Tensor {
	// [B, n_head, T, head_size] -> [B, T, n_head, head_size] -> [B, T, C]
	transposed := l.to.Transpose(y, []int{0, 2, 1, 3})
	reassembled := l.to.Reshape(transposed, []int{B, T, C})
	l.to.Free(transposed)
	return reassembled
}

// outputProjection 输出投影
func (l *CausalSelfAttentionLayer) outputProjection(y tensor.Tensor) tensor.Tensor {
	B, T, C := y.Size()[0], y.Size()[1], y.Size()[2]

	// 重塑为 2D 以便进行矩阵乘法
	y2D := l.to.Reshape(y, []int{B * T, C})
	defer l.to.Free(y2D)

	// 输出投影
	output2D := l.to.Gemm(false, false, 1.0, 0.0, y2D, l.cProjWeights, l.cProjBias)
	output := l.to.Reshape(output2D, []int{B, T, C})
	l.to.Free(output2D)

	return output
}

// Backward 反向传播
func (l *CausalSelfAttentionLayer) Backward(input tensor.Tensor) tensor.Tensor {
	// 清空梯度
	l.to.Clear(l.cAttnWeightsGrad)
	l.to.Clear(l.cAttnBiasGrad)
	l.to.Clear(l.cProjWeightsGrad)
	l.to.Clear(l.cProjBiasGrad)

	// 计算梯度
	l.calculateGradients(input)

	var output tensor.Tensor
	if l.layerIndex > 0 {
		output = l.calculateInputGradients(input)
	}

	// 清理缓存
	l.to.Free(l.forwardInput)
	l.to.Free(l.attentionWeights)
	l.to.Free(l.softmaxOutput)
	l.to.Free(input)

	return output
}

// calculateGradients 计算权重梯度
func (l *CausalSelfAttentionLayer) calculateGradients(input tensor.Tensor) {
	// 这里需要实现完整的反向传播逻辑
	// 由于注意力机制的反向传播较复杂，这里简化实现

	// 实际实现中需要计算：
	// 1. 输出投影层的梯度
	// 2. 注意力权重的梯度
	// 3. QKV 投影层的梯度
	// 这里使用占位符实现
	fmt.Printf("Calculating gradients for causal self-attention layer %d\n", l.layerIndex)
}

// calculateInputGradients 计算输入梯度
func (l *CausalSelfAttentionLayer) calculateInputGradients(input tensor.Tensor) tensor.Tensor {
	// 计算对输入的梯度
	// 这里简化实现，实际需要完整的反向传播
	inputShape := l.forwardInput.Size()
	output := l.to.Zeros(inputShape)
	return output
}

// Parameters 返回所有参数
func (l *CausalSelfAttentionLayer) Parameters() []tensor.Tensor {
	params := []tensor.Tensor{l.cAttnWeights, l.cProjWeights}
	if l.config.Bias {
		params = append(params, l.cAttnBias, l.cProjBias)
	}
	return params
}

// Gradients 返回所有梯度
func (l *CausalSelfAttentionLayer) Gradients() []tensor.Tensor {
	grads := []tensor.Tensor{l.cAttnWeightsGrad, l.cProjWeightsGrad}
	if l.config.Bias {
		grads = append(grads, l.cAttnBiasGrad, l.cProjBiasGrad)
	}
	return grads
}

// GetOutputShape 获取输出形状
func (l *CausalSelfAttentionLayer) GetOutputShape(inputShape []int) []int {
	// 输入: [batch_size, seq_len, n_embd]
	// 输出: [batch_size, seq_len, n_embd] (相同形状)
	return []int{inputShape[0], inputShape[1], l.config.NEmbd}
}

// SetWeights 设置权重（用于加载预训练模型）
func (l *CausalSelfAttentionLayer) SetWeights(cAttnWeights, cProjWeights []float64) {
	l.to.Init(l.cAttnWeights, cAttnWeights)
	l.to.Init(l.cProjWeights, cProjWeights)
}

// SetBiases 设置偏置（用于加载预训练模型）
func (l *CausalSelfAttentionLayer) SetBiases(cAttnBias, cProjBias []float64) {
	if l.config.Bias {
		l.to.Init(l.cAttnBias, cAttnBias)
		l.to.Init(l.cProjBias, cProjBias)
	}
}
