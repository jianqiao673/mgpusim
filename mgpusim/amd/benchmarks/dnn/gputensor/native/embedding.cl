// Embedding 前向传播 kernel
__kernel void embedding_forward(
    __global const int* input_indices,
    __global const float* weight,
    __global float* output,
    const int vocab_size,
    const int embedding_dim,
    const int batch_size,
    const int seq_len,
    const int padding_idx
) {
    int gid = get_global_id(0);
    
    if (gid >= batch_size * seq_len) {
        return;
    }
    
    int index = input_indices[gid];
    
    // 处理 padding 索引
    if (index == padding_idx) {
        for (int i = 0; i < embedding_dim; i++) {
            output[gid * embedding_dim + i] = 0.0f;
        }
        return;
    }
    
    // 边界检查
    if (index < 0 || index >= vocab_size) {
        // 索引越界，设置为0
        for (int i = 0; i < embedding_dim; i++) {
            output[gid * embedding_dim + i] = 0.0f;
        }
        return;
    }
    
    // 从权重矩阵中复制对应的嵌入向量
    for (int i = 0; i < embedding_dim; i++) {
        output[gid * embedding_dim + i] = weight[index * embedding_dim + i];
    }
}

// Embedding 权重梯度计算 kernel
__kernel void embedding_backward_weight(
    __global const int* input_indices,
    __global const float* grad_output,
    __global float* grad_weight,
    const int vocab_size,
    const int embedding_dim,
    const int batch_size,
    const int seq_len,
    const int padding_idx,
    const float scale
) {
    int word_id = get_global_id(0);
    
    if (word_id >= vocab_size) {
        return;
    }
    
    // 为每个词汇初始化梯度为0
    for (int i = 0; i < embedding_dim; i++) {
        grad_weight[word_id * embedding_dim + i] = 0.0f;
    }
    
    // 累加所有出现该词汇的位置的梯度
    for (int pos = 0; pos < batch_size * seq_len; pos++) {
        int index = input_indices[pos];
        
        if (index == word_id && index != padding_idx) {
            for (int i = 0; i < embedding_dim; i++) {
                grad_weight[word_id * embedding_dim + i] += 
                    grad_output[pos * embedding_dim + i] * scale;
            }
        }
    }
}