package optimization

import (
	"log"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// AdamW 优化器
type AdamW struct {
	to tensor.Operator

	LR          float64
	WeightDecay float64
	Beta1       float64
	Beta2       float64
	Eps         float64

	historyV map[Layer]tensor.Tensor
	historyS map[Layer]tensor.Tensor
}

// NewAdamW 创建 AdamW
func NewAdamW(to tensor.Operator, lr, wd float64) *AdamW {
	return &AdamW{
		to:          to,
		LR:          lr,
		WeightDecay: wd,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		historyV:    make(map[Layer]tensor.Tensor),
		historyS:    make(map[Layer]tensor.Tensor),
	}
}

// UpdateParameters 更新参数
func (r *AdamW) UpdateParameters(layer Layer) {
	params := layer.Parameters()
	grads := layer.Gradients()

	if params == nil || grads == nil {
		return
	}

	v := r.historyV[layer]
	s, found := r.historyS[layer]

	if !found {
		// 初始化历史 v/s
		v = r.to.Clone(grads)
		r.historyV[layer] = v

		s = r.to.Clone(grads)
		sSquare := r.to.ElementWiseMul(s, s)
		r.to.Free(s)
		s = sSquare
		r.historyS[layer] = s
	}

	// 权重衰减
	if r.WeightDecay > 0 {
		//params = r.to.MulScalar(params, 1-r.LR*r.WeightDecay)
	}

	// 使用 tensor.Operator 自带 Adam 方法更新参数
	r.to.Adam(params, grads, v, s, r.Beta1, r.Beta2, r.LR)
}

// LazyUpdateParameters 延迟更新（只更新 v/s，不更新参数）
func (r *AdamW) LazyUpdateParameters(layer Layer) {
	params := layer.Parameters()
	grads := layer.Gradients()

	if params == nil || grads == nil {
		return
	}

	v := r.historyV[layer]
	s, found := r.historyS[layer]

	if !found {
		// 初始化历史 v/s
		v = r.to.LazyClone(grads)
		r.historyV[layer] = v

		s = r.to.LazyClone(grads)
		sSquare := r.to.LazyElementWiseMul(s, s)
		r.to.Free(s)
		s = sSquare
		r.historyS[layer] = s

		log.Printf("[LazyUpdateParameters] historyV: 0x%x, historyS: 0x%x\n", v, s)
	}

	// 使用 tensor.Operator 自带 LazyAdam 方法
	r.to.LazyAdam(params, grads, v, s, r.Beta1, r.Beta2, r.LR)
}
