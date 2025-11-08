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
	BlockSize int  // 序列长度（用于 mask）
}

// CausalSelfAttentionLayer 因果自注意力层
type CausalSelfAttentionLayer struct {
	layerIndex int
	to         tensor.Operator
	config     CausalSelfAttentionConfig

	// 权重参数
	cAttnWeights tensor.Tensor // [n_embd, 3*n_embd]
	cAttnBias    tensor.Tensor // [3*n_embd] (optional)
	cProjWeights tensor.Tensor // [n_embd, n_embd]
	cProjBias    tensor.Tensor // [n_embd] (optional)

	// 梯度张量
	cAttnWeightsGrad tensor.Tensor
	cAttnBiasGrad    tensor.Tensor
	cProjWeightsGrad tensor.Tensor
	cProjBiasGrad    tensor.Tensor

	// 前向传播缓存
	forwardInput     tensor.Tensor
	attentionWeights tensor.Tensor
	softmaxOutput    tensor.Tensor
	qkv              tensor.Tensor
	q, k, v          tensor.Tensor
}

// NewCausalSelfAttentionLayer 创建新的因果自注意力层
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

	// 参数分配
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

// Randomize 随机初始化参数
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

// Forward 前向传播
func (l *CausalSelfAttentionLayer) Forward(input tensor.Tensor) tensor.Tensor {
	// 验证输入形状
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

	// 保存输入用于反向传播
	l.forwardInput = l.to.Clone(input)

	// === 1) QKV 投影 ===
	in2D := l.to.Reshape(input, []int{B * T, C})

	// 创建正确形状的输出矩阵
	outputShape := []int{B * T, 3 * C}
	outputMatrix := l.to.Zeros(outputShape)

	// 处理偏置
	if l.config.Bias && l.cAttnBias != nil {
		// 将1D偏置重塑为2D并与输出矩阵相加
		bias2D := l.to.Reshape(l.cAttnBias, []int{1, 3 * C})
		// 复制偏置以匹配输出形状
		repeatedBias := l.to.Repeat(bias2D, B*T)
		outputMatrix = l.to.ScaleAdd(1.0, 1.0, outputMatrix, repeatedBias)
		//l.to.Free(bias2D)
		//l.to.Free(repeatedBias)
	}

	qkv2D := l.to.Gemm(false, false, 1.0, 1.0, in2D, l.cAttnWeights, outputMatrix)
	//l.to.Free(in2D)
	//l.to.Free(outputMatrix)

	l.qkv = l.to.Reshape(qkv2D, []int{B, T, 3 * C})
	//l.to.Free(qkv2D)

	// === 2) QKV 分割 ===
	l.q, l.k, l.v = l.splitQKV(l.qkv, B, T, C)

	// === 3) reshape to multi-head ===
	q := l.reshapeToMultiHead(l.q, B, T, headSize)
	k := l.reshapeToMultiHead(l.k, B, T, headSize)
	v := l.reshapeToMultiHead(l.v, B, T, headSize)

	// === 4) 计算注意力 scores ===
	att := l.computeAttentionScores(q, k, B, T, headSize)
	//l.to.Free(q)
	//l.to.Free(k)

	// === 5) causal mask ===
	attMasked := l.applyCausalMask(att, B, T)
	//l.to.Free(att)

	// === 6) softmax (确保是2D输入) ===
	attMaskedSize := attMasked.Size()
	totalElements := B * l.config.NHead * T * T
	attMasked2D := l.to.Reshape(attMasked, []int{totalElements / T, T})

	softmaxOut2D := l.to.Softmax(attMasked2D)
	softmaxOut := l.to.Reshape(softmaxOut2D, attMaskedSize)

	l.softmaxOutput = l.to.Clone(softmaxOut)

	// 释放临时张量
	//l.to.Free(attMasked)
	//l.to.Free(attMasked2D)
	//l.to.Free(softmaxOut2D)

	// === 7) attention-weighted sum ===
	y := l.applyAttention(softmaxOut, v, B, T, headSize)
	//l.to.Free(v)
	//l.to.Free(softmaxOut)

	// === 8) reassemble heads ===
	y = l.reassembleHeads(y, B, T, C)

	// === 9) 输出投影 ===
	y2D := l.to.Reshape(y, []int{B * T, C})

	// 创建正确形状的输出矩阵
	projOutputShape := []int{B * T, C}
	projOutputMatrix := l.to.Zeros(projOutputShape)

	// 处理输出投影的偏置
	if l.config.Bias && l.cProjBias != nil {
		// 将1D偏置重塑为2D并与输出矩阵相加
		projBias2D := l.to.Reshape(l.cProjBias, []int{1, C})
		// 复制偏置以匹配输出形状
		projRepeatedBias := l.to.Repeat(projBias2D, B*T)
		projOutputMatrix = l.to.ScaleAdd(1.0, 1.0, projOutputMatrix, projRepeatedBias)
		//l.to.Free(projBias2D)
		//l.to.Free(projRepeatedBias)
	}

	out2D := l.to.Gemm(false, false, 1.0, 1.0, y2D, l.cProjWeights, projOutputMatrix)
	//l.to.Free(y2D)
	//l.to.Free(y)
	//l.to.Free(projOutputMatrix)

	out := l.to.Reshape(out2D, []int{B, T, C})
	//l.to.Free(out2D)

	return out
}

// splitQKV 将 qkv ([B,T,3*C]) 拆分为 q, k, v ([B,T,C] each)
func (l *CausalSelfAttentionLayer) splitQKV(qkv tensor.Tensor, B, T, C int) (q, k, v tensor.Tensor) {
	// 使用切片操作避免数据拷贝
	totalElements := B * T * 3 * C

	// 切片获取q, k, v
	q = l.to.Slice(qkv, 0, B*T*C)
	k = l.to.Slice(qkv, B*T*C, 2*B*T*C)
	v = l.to.Slice(qkv, 2*B*T*C, totalElements)

	// 重塑形状
	q = l.to.Reshape(q, []int{B, T, C})
	k = l.to.Reshape(k, []int{B, T, C})
	v = l.to.Reshape(v, []int{B, T, C})

	return
}

// reshapeToMultiHead 重塑为多头格式
func (l *CausalSelfAttentionLayer) reshapeToMultiHead(t tensor.Tensor, B, T, headSize int) tensor.Tensor {
	r1 := l.to.Reshape(t, []int{B, T, l.config.NHead, headSize})
	out := l.to.Transpose(r1, []int{0, 2, 1, 3})
	//l.to.Free(r1)
	//l.to.Free(t)
	return out
}

// computeAttentionScores 计算注意力分数
func (l *CausalSelfAttentionLayer) computeAttentionScores(q, k tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// 转置 k
	kT := l.to.Transpose(k, []int{0, 1, 3, 2})
	defer l.to.Free(kT)

	// 批量矩阵乘法
	att := l.batchMatMul(q, kT, B, l.config.NHead, T, headSize, T)

	// 缩放
	scale := 1.0 / math.Sqrt(float64(headSize))
	return l.scaleTensor(att, scale)
}

// batchMatMul 批量矩阵乘法实现
func (l *CausalSelfAttentionLayer) batchMatMul(a, b tensor.Tensor, B, nHead, T, M, N int) tensor.Tensor {
	aData := a.Vector()
	bData := b.Vector()
	resultData := make([]float64, B*nHead*T*N)

	// 对每个batch和head进行循环
	for bIdx := 0; bIdx < B; bIdx++ {
		for h := 0; h < nHead; h++ {
			// 计算当前batch和head的起始索引
			aBase := (bIdx*nHead + h) * T * M
			bBase := (bIdx*nHead + h) * M * N
			rBase := (bIdx*nHead + h) * T * N

			// 计算矩阵乘法: [T, M] @ [M, N] = [T, N]
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

	// 释放输入张量
	//l.to.Free(a)
	//l.to.Free(b)

	return l.to.CreateWithData(resultData, []int{B, nHead, T, N}, "batch_matmul")
}

// scaleTensor 张量缩放
func (l *CausalSelfAttentionLayer) scaleTensor(t tensor.Tensor, scale float64) tensor.Tensor {
	data := t.Vector()
	result := make([]float64, len(data))
	for i := range data {
		result[i] = data[i] * scale
	}
	//l.to.Free(t)
	return l.to.CreateWithData(result, t.Size(), "scaled")
}

// applyCausalMask 应用因果掩码
func (l *CausalSelfAttentionLayer) applyCausalMask(att tensor.Tensor, B, T int) tensor.Tensor {
	attData := att.Vector()
	result := make([]float64, len(attData))
	copy(result, attData)

	attSize := att.Size()
	nHead := attSize[1]

	// 直接在数据上应用掩码，避免创建完整的mask矩阵
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

	//l.to.Free(att)
	return l.to.CreateWithData(result, attSize, "masked_att")
}

// applyAttention 应用注意力权重
func (l *CausalSelfAttentionLayer) applyAttention(att, v tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// 使用批量矩阵乘法
	return l.batchMatMul(att, v, B, l.config.NHead, T, T, headSize)
}

// reassembleHeads 重组多头
func (l *CausalSelfAttentionLayer) reassembleHeads(y tensor.Tensor, B, T, C int) tensor.Tensor {
	nHead := l.config.NHead
	headDim := C / nHead
	// 1) reshape [B, nHead, T, headDim] -> [B, T, nHead*headDim]
	reshaped := l.to.Reshape(y, []int{B, T, nHead * headDim})

	// 释放原张量
	//l.to.Free(y)

	return reshaped
}

// Backward 反向传播（简化实现）
func (l *CausalSelfAttentionLayer) Backward(gradOutput tensor.Tensor) tensor.Tensor {
	B, T, C := gradOutput.Size()[0], gradOutput.Size()[1], gradOutput.Size()[2]

	// 初始化 cProjWeightsGrad
	if l.cProjWeightsGrad == nil {
		l.cProjWeightsGrad = l.to.Zeros(l.cProjWeights.Size())
	}

	// reshape gradOutput 和 forwardInput 为 2D
	gradOutput2D := l.to.Reshape(gradOutput, []int{B * T, C})
	y2D := l.to.Reshape(l.forwardInput, []int{B * T, C})

	// 计算 c_proj 权重梯度
	l.cProjWeightsGrad = l.to.Gemm(true, false, 1.0, 1.0, y2D, gradOutput2D, l.cProjWeightsGrad)

	// 输入梯度
	gradInput2D := l.to.Zeros([]int{B * T, C})
	gradInput2D = l.to.Gemm(false, true, 1.0, 0.0, gradOutput2D, l.cProjWeights, gradInput2D)
	gradInput := l.to.Reshape(gradInput2D, []int{B, T, C})

	// 清理临时张量
	//l.to.Free(gradOutput2D)
	//l.to.Free(y2D)
	//l.to.Free(gradInput2D)

	l.cleanupForwardCache()

	return gradInput
}

// cleanupForwardCache 清理前向传播缓存
func (l *CausalSelfAttentionLayer) cleanupForwardCache() {
	if l.forwardInput != nil {
		//l.to.Free(l.forwardInput)
		l.forwardInput = nil
	}
	if l.softmaxOutput != nil {
		//l.to.Free(l.softmaxOutput)
		l.softmaxOutput = nil
	}
	if l.qkv != nil {
		//l.to.Free(l.qkv)
		l.qkv = nil
	}
	if l.q != nil {
		//l.to.Free(l.q)
		l.q = nil
	}
	if l.k != nil {
		//l.to.Free(l.k)
		l.k = nil
	}
	if l.v != nil {
		//l.to.Free(l.v)
		l.v = nil
	}
}

// Parameters 返回参数集合
func (l *CausalSelfAttentionLayer) Parameters() []tensor.Tensor {
	params := []tensor.Tensor{l.cAttnWeights, l.cProjWeights}
	if l.config.Bias {
		params = append(params, l.cAttnBias, l.cProjBias)
	}
	return params
}

// Gradients 返回对应梯度集合
func (l *CausalSelfAttentionLayer) Gradients() []tensor.Tensor {
	grads := []tensor.Tensor{l.cAttnWeightsGrad, l.cProjWeightsGrad}
	if l.config.Bias {
		grads = append(grads, l.cAttnBiasGrad, l.cProjBiasGrad)
	}
	return grads
}

// GetOutputShape 获取输出形状
func (l *CausalSelfAttentionLayer) GetOutputShape(inputShape []int) []int {
	return []int{inputShape[0], inputShape[1], l.config.NEmbd}
}

// SetWeights 设置权重
func (l *CausalSelfAttentionLayer) SetWeights(cAttnWeights, cProjWeights []float64) {
	l.to.Init(l.cAttnWeights, cAttnWeights)
	l.to.Init(l.cProjWeights, cProjWeights)
}

// SetBiases 设置偏置
func (l *CausalSelfAttentionLayer) SetBiases(cAttnBias, cProjBias []float64) {
	if l.config.Bias {
		l.to.Init(l.cAttnBias, cAttnBias)
		l.to.Init(l.cProjBias, cProjBias)
	}
}

// Close 释放所有资源
func (l *CausalSelfAttentionLayer) Close() {
	l.cleanupForwardCache()

	// 释放参数和梯度
	params := l.Parameters()
	grads := l.Gradients()

	for _, p := range params {
		if p != nil {
			//l.to.Free(p)
		}
	}
	for _, g := range grads {
		if g != nil {
			//l.to.Free(g)
		}
	}
}
