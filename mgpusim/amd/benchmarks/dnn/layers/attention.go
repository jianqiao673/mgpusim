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

	// 权重参数（参数存储布局与 transformer 常见实现一致）
	cAttnWeights tensor.Tensor // [n_embd, 3*n_embd] or flattened accordingly
	cAttnBias    tensor.Tensor // [3*n_embd] (optional)
	cProjWeights tensor.Tensor // [n_embd, n_embd]
	cProjBias    tensor.Tensor // [n_embd] (optional)

	// 梯度张量（与参数相对应）
	cAttnWeightsGrad tensor.Tensor
	cAttnBiasGrad    tensor.Tensor
	cProjWeightsGrad tensor.Tensor
	cProjBiasGrad    tensor.Tensor

	// 前向传播缓存（用于反向或调试）
	forwardInput     tensor.Tensor // clone of input indices/embedding (kept briefly)
	attentionWeights tensor.Tensor // softmax(att) （保留用于检查/反向）
	softmaxOutput    tensor.Tensor // clone of attentionWeights（如果需要）
}

// NewCausalSelfAttentionLayer 创建新的因果自注意力层
func NewCausalSelfAttentionLayer(
	index int,
	to tensor.Operator,
	config CausalSelfAttentionConfig,
) *CausalSelfAttentionLayer {
	// 检查配置
	if config.NEmbd%config.NHead != 0 {
		panic(fmt.Sprintf("n_embd (%d) must be divisible by n_head (%d)",
			config.NEmbd, config.NHead))
	}

	l := &CausalSelfAttentionLayer{
		layerIndex: index,
		to:         to,
		config:     config,
	}

	// 参数分配。注意：Create 的参数是 shape 列表（不同实现可能处理不同）
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

// Randomize 随机初始化参数（Xavier-like）
func (l *CausalSelfAttentionLayer) Randomize() {
	// Xavier scale
	xavier := math.Sqrt(2.0 / float64(l.config.NEmbd))

	// c_attn weights: shape n_embd x 3*n_embd => length n_embd * 3*n_embd
	n1 := l.config.NEmbd * 3 * l.config.NEmbd
	cAttnData := make([]float64, n1)
	for i := range cAttnData {
		cAttnData[i] = (rand.Float64()*2 - 1) * xavier
	}
	l.to.Init(l.cAttnWeights, cAttnData)

	// c_proj weights: n_embd x n_embd
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

// Forward 前向传播（尽量使用 Clone / Reshape / Free 以减少峰值内存）
func (l *CausalSelfAttentionLayer) Forward(input tensor.Tensor) tensor.Tensor {
	// 保留一份输入（用于后向或验证），但尽可能短期保留
	l.forwardInput = l.to.Clone(input)

	// Input shape: [B, T, C]
	B := input.Size()[0]
	T := input.Size()[1]
	C := input.Size()[2]
	headSize := C / l.config.NHead

	// === 1) QKV 投影 ===
	// 为了使用已有的 Gemm（2D），先 reshape 为 [B*T, C]
	in2D := l.to.Reshape(input, []int{B * T, C})

	// qkv2D shape: [B*T, 3*C]
	qkv2D := l.to.Gemm(false, false, 1.0, 0.0, in2D, l.cAttnWeights, l.cAttnBias)

	// 释放 reshape 的临时 in2D 以及原始 input（我们已 clone forwardInput）
	l.to.Free(in2D)
	l.to.Free(input)

	// reshape 回 [B, T, 3*C]
	qkv := l.to.Reshape(qkv2D, []int{B, T, 3 * C})
	l.to.Free(qkv2D)

	// === 2) 分割 Q,K,V ===
	// 这里为了兼容通用 operator，我们用创建拷贝的方式分割（某些 GPUOperator 可直接支持按最后一维 split）
	q, k, v := l.splitQKV(qkv, B, T, C)
	l.to.Free(qkv) // 释放原始 qkv

	// === 3) 重塑为多头格式 ===
	// q,k,v: [B, T, C] -> reshape to [B, T, n_head, headSize] -> transpose to [B, n_head, T, headSize]
	q = l.reshapeToMultiHead(q, B, T, headSize)
	k = l.reshapeToMultiHead(k, B, T, headSize)
	v = l.reshapeToMultiHead(v, B, T, headSize)

	// === 4) 计算注意力分数 ===
	att := l.computeAttentionScores(q, k, B, T, headSize)

	// q,k 可释放（不再需要）
	l.to.Free(q)
	l.to.Free(k)

	// === 5) apply causal mask ===
	attMasked := l.applyCausalMask(att, T)
	l.to.Free(att)

	// === 6) softmax ===
	l.attentionWeights = l.to.Softmax(attMasked)
	// 备份 softmax 结果（如果需要在 backward 用到）
	l.softmaxOutput = l.to.Clone(l.attentionWeights) // clone 一份便于后向
	l.to.Free(attMasked)

	// === 7) attention-weighted sum: att @ v ===
	y := l.applyAttention(l.attentionWeights, v, B, T, headSize)
	l.to.Free(v)
	l.to.Free(l.attentionWeights) // 已经 clone 了一份放在 softmaxOutput

	// === 8) reassemble heads ===
	y = l.reassembleHeads(y, B, T, C)

	// === 9) output projection ===
	out := l.outputProjection(y)
	l.to.Free(y)

	// 返回最终结果（不释放 forwardInput / softmaxOutput，这些由 caller/backward 管理）
	return out
}

// splitQKV 将 qkv ([B,T,3*C]) 拆分为 q, k, v ([B,T,C] each)
// 说明：通用实现会将数据复制到新张量中（兼容 CPUOperator）。若 GPUOperator 提供按最后维度切片而不拷贝的能力，可以替换这里的实现以节省内存。
func (l *CausalSelfAttentionLayer) splitQKV(qkv tensor.Tensor, B, T, C int) (q, k, v tensor.Tensor) {
	// 将 qkv 转为扁平向量并拷贝为三个独立张量
	qkvData := qkv.Vector() // flatten data

	qData := make([]float64, B*T*C)
	kData := make([]float64, B*T*C)
	vData := make([]float64, B*T*C)

	for i := 0; i < B*T; i++ {
		// 每个位置有 3*C 的数据： [i*3*C : i*3*C + 3*C)
		base := i * 3 * C
		copy(qData[i*C:(i+1)*C], qkvData[base:base+C])
		copy(kData[i*C:(i+1)*C], qkvData[base+C:base+2*C])
		copy(vData[i*C:(i+1)*C], qkvData[base+2*C:base+3*C])
	}

	q = l.to.CreateWithData(qData, []int{B, T, C}, "q")
	k = l.to.CreateWithData(kData, []int{B, T, C}, "k")
	v = l.to.CreateWithData(vData, []int{B, T, C}, "v")
	return
}

// reshapeToMultiHead: [B,T,C] -> [B,T,n_head,headSize] -> transpose -> [B,n_head,T,headSize]
func (l *CausalSelfAttentionLayer) reshapeToMultiHead(t tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// reshape -> [B, T, n_head, headSize]
	r1 := l.to.Reshape(t, []int{B, T, l.config.NHead, headSize})
	// transpose (0,2,1,3) -> [B, n_head, T, headSize]
	out := l.to.Transpose(r1, []int{0, 2, 1, 3})
	l.to.Free(r1)
	l.to.Free(t)
	return out
}

// computeAttentionScores: q @ k^T / sqrt(headSize) -> [B, n_head, T, T]
func (l *CausalSelfAttentionLayer) computeAttentionScores(q, k tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// kT: [B, n_head, headSize, T]
	kT := l.to.Transpose(k, []int{0, 1, 3, 2})
	defer l.to.Free(kT)

	// batch matmul: [B, n_head, T, headSize] @ [B, n_head, headSize, T] -> [B, n_head, T, T]
	att := l.batchMatMul4D(q, kT)

	// scale
	scale := 1.0 / math.Sqrt(float64(headSize))
	scaled := l.scaleTensor(att, scale)
	l.to.Free(att)
	return scaled
}

// batchMatMul4D: 使用 reshape->Gemm->reshape 的方式计算批量 4D 矩阵乘法
// a: [B, nHead, T, M], b: [B, nHead, M, N] => out: [B, nHead, T, N]
func (l *CausalSelfAttentionLayer) batchMatMul4D(a, b tensor.Tensor) tensor.Tensor {
	aSize := a.Size()
	B, nHead, T, M := aSize[0], aSize[1], aSize[2], aSize[3]
	// b size: [B, nHead, M, N]
	N := b.Size()[3]
	// 逐 batch/head 计算
	outSlices := make([]tensor.Tensor, 0, B*nHead)
	for bIdx := 0; bIdx < B; bIdx++ {
		for h := 0; h < nHead; h++ {
			// extract a_sub: [T, M] from a
			a_sub := l.to.Slice(a, ((bIdx*nHead+h)*T)*M, ((bIdx*nHead+h+1)*T)*M) // flattened slice
			a_sub2 := l.to.Reshape(a_sub, []int{T, M})
			// extract b_sub: [M, N] from b
			b_sub := l.to.Slice(b, ((bIdx*nHead+h)*M)*N, ((bIdx*nHead+h+1)*M)*N)
			b_sub2 := l.to.Reshape(b_sub, []int{M, N})

			// out_sub = a_sub2 @ b_sub2 -> [T, N]
			zero := l.to.Zeros([]int{T, N})
			out_sub := l.to.Gemm(false, false, 1.0, 0.0, a_sub2, b_sub2, zero)

			// reshape to [1,1,T,N] later合并
			outSlices = append(outSlices, l.to.Reshape(out_sub, []int{1, 1, T, N}))

			// free intermediates
			l.to.Free(a_sub)
			l.to.Free(a_sub2)
			l.to.Free(b_sub)
			l.to.Free(b_sub2)
			l.to.Free(zero)
			l.to.Free(out_sub)
		}
	}

	// concat outSlices into [B, nHead, T, N]
	// 将 slices 按顺序合并（这里用最简单的方法：收集各 slice 的 vector，拼接然后 CreateWithData）
	totalElems := B * nHead * T * N
	all := make([]float64, 0, totalElems)
	for _, s := range outSlices {
		all = append(all, s.Vector()...)
		l.to.Free(s)
	}
	out := l.to.CreateWithData(all, []int{B, nHead, T, N}, "att_batch")
	return out
}

// scaleTensor 元素缩放
func (l *CausalSelfAttentionLayer) scaleTensor(t tensor.Tensor, scale float64) tensor.Tensor {
	data := t.Vector()
	out := make([]float64, len(data))
	for i := range data {
		out[i] = data[i] * scale
	}
	return l.to.CreateWithData(out, t.Size(), "scaled")
}

// applyCausalMask: 使用下三角 mask，将上三角位置设为 -inf (用一个很小的负数)
func (l *CausalSelfAttentionLayer) applyCausalMask(att tensor.Tensor, T int) tensor.Tensor {
	// 创建 causal mask [1,1,T,T]
	maskData := make([]float64, T*T)
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			if j > i {
				maskData[i*T+j] = -1e9
			} else {
				maskData[i*T+j] = 0.0
			}
		}
	}
	mask := l.to.CreateWithData(maskData, []int{1, 1, T, T}, "causal_mask")
	// 广播并相加
	attSize := att.Size()
	B, nHead := attSize[0], attSize[1]
	broadcast := l.broadcastMask(mask, B, nHead, T)
	l.to.Free(mask)
	out := l.addTensors(att, broadcast)
	l.to.Free(broadcast)
	return out
}

// broadcastMask: 把 mask 扩展到 [B, nHead, T, T]
func (l *CausalSelfAttentionLayer) broadcastMask(mask tensor.Tensor, B, nHead, T int) tensor.Tensor {
	m := mask.Vector() // length T*T
	outData := make([]float64, B*nHead*T*T)
	for b := 0; b < B; b++ {
		for h := 0; h < nHead; h++ {
			for i := 0; i < T; i++ {
				for j := 0; j < T; j++ {
					idx := ((b*nHead+h)*T+i)*T + j
					outData[idx] = m[i*T+j]
				}
			}
		}
	}
	return l.to.CreateWithData(outData, []int{B, nHead, T, T}, "broadcast_mask")
}

// addTensors 元素相加
func (l *CausalSelfAttentionLayer) addTensors(a, b tensor.Tensor) tensor.Tensor {
	ad := a.Vector()
	bd := b.Vector()
	res := make([]float64, len(ad))
	for i := range ad {
		res[i] = ad[i] + bd[i]
	}
	return l.to.CreateWithData(res, a.Size(), "added")
}

// applyAttention: att [B,n_head,T,T] @ v [B,n_head,T,headSize] -> [B,n_head,T,headSize]
func (l *CausalSelfAttentionLayer) applyAttention(att, v tensor.Tensor, B, T, headSize int) tensor.Tensor {
	// reuse batchMatMul4D (att is [B,nHead,T,T], v is [B,nHead,T,headSize])
	// Convert v to shape [B,nHead, T, headSize] already is.
	// For multiplication, consider v as [B,nHead,T,headSize] and compute att @ v
	// we transpose v to [B,nHead, headSize, T]? actually we need att @ v: (T x T) @ (T x headSize) -> (T x headSize)

	// We can implement by looping each (B, h) block (similar to batchMatMul4D approach)
	Bn := att.Size()[0]
	nHead := att.Size()[1]
	N := headSize

	outSlices := make([]tensor.Tensor, 0, Bn*nHead)
	for bIdx := 0; bIdx < Bn; bIdx++ {
		for h := 0; h < nHead; h++ {
			att_sub := l.to.Slice(att, ((bIdx*nHead+h)*T)*T, ((bIdx*nHead+h+1)*T)*T)
			att2 := l.to.Reshape(att_sub, []int{T, T})
			v_sub := l.to.Slice(v, ((bIdx*nHead+h)*T)*N, ((bIdx*nHead+h+1)*T)*N)
			v2 := l.to.Reshape(v_sub, []int{T, N})

			zero := l.to.Zeros([]int{T, N})
			out_sub := l.to.Gemm(false, false, 1.0, 0.0, att2, v2, zero)
			outSlices = append(outSlices, l.to.Reshape(out_sub, []int{1, 1, T, N}))

			// free temporaries
			l.to.Free(att_sub)
			l.to.Free(att2)
			l.to.Free(v_sub)
			l.to.Free(v2)
			l.to.Free(zero)
			l.to.Free(out_sub)
		}
	}

	// 合并
	totalElems := Bn * nHead * T * N
	all := make([]float64, 0, totalElems)
	for _, s := range outSlices {
		all = append(all, s.Vector()...)
		l.to.Free(s)
	}
	out := l.to.CreateWithData(all, []int{Bn, nHead, T, N}, "att_v")
	return out
}

// reassembleHeads: [B,n_head,T,headSize] -> transpose -> [B,T,C]
func (l *CausalSelfAttentionLayer) reassembleHeads(y tensor.Tensor, B, T, C int) tensor.Tensor {
	// transpose [B,T,n_head,headSize] desired; currently y is [B,n_head,T,headSize]
	transposed := l.to.Transpose(y, []int{0, 2, 1, 3})
	reshaped := l.to.Reshape(transposed, []int{B, T, C})
	l.to.Free(transposed)
	l.to.Free(y)
	return reshaped
}

// outputProjection: y [B,T,C] -> reshape and gemm with cProjWeights
func (l *CausalSelfAttentionLayer) outputProjection(y tensor.Tensor) tensor.Tensor {
	B, T, C := y.Size()[0], y.Size()[1], y.Size()[2]
	y2D := l.to.Reshape(y, []int{B * T, C})
	out2D := l.to.Gemm(false, false, 1.0, 0.0, y2D, l.cProjWeights, l.cProjBias)
	l.to.Free(y2D)
	out := l.to.Reshape(out2D, []int{B, T, C})
	l.to.Free(out2D)
	return out
}

// Backward: 简化实现（占位），清理缓存并返回输入梯度占位
func (l *CausalSelfAttentionLayer) Backward(input tensor.Tensor) tensor.Tensor {
	// 清空梯度张量（占位）
	l.to.Clear(l.cAttnWeightsGrad)
	l.to.Clear(l.cProjWeightsGrad)
	if l.config.Bias {
		l.to.Clear(l.cAttnBiasGrad)
		l.to.Clear(l.cProjBiasGrad)
	}

	// placeholder: 复杂反向传播在此省略（需要完整实现 QKV、softmax、GEMM 的反向）
	fmt.Printf("[CausalSelfAttentionLayer.Backward] layer %d: placeholder gradient computation\n", l.layerIndex)

	// 返回与 forwardInput sameshape 的 zeros 作为输入梯度
	inShape := l.forwardInput.Size()
	out := l.to.Zeros(inShape)

	// 释放缓存
	l.to.Free(l.forwardInput)
	l.to.Free(l.softmaxOutput)
	l.to.Free(input)

	return out
}

// Parameters 返回参数集合（便于优化器处理）
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
	// 输入: [batch, seq, n_embd] -> 输出: 相同形状
	return []int{inputShape[0], inputShape[1], l.config.NEmbd}
}

// SetWeights / SetBiases 用于加载预训练参数（可选）
func (l *CausalSelfAttentionLayer) SetWeights(cAttnWeights, cProjWeights []float64) {
	l.to.Init(l.cAttnWeights, cAttnWeights)
	l.to.Init(l.cProjWeights, cProjWeights)
}
func (l *CausalSelfAttentionLayer) SetBiases(cAttnBias, cProjBias []float64) {
	if l.config.Bias {
		l.to.Init(l.cAttnBias, cAttnBias)
		l.to.Init(l.cProjBias, cProjBias)
	}
}
