// simplified_attention.cl
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define SOFTMAX_MIN -1e20f
#define MAX_SEQ_LEN 256  // 限制序列长度

// 简化的统一注意力内核 - 避免使用局部内存
__kernel void attention(
    __global const float* x,           // [B, T, C]
    __global const float* c_attn,      // [C, 3*C]
    __global const float* c_proj,      // [C, C]
    __global float* output,            // [B, T, C]
    const int B,
    const int T,
    const int C,
    const int nHead
) {
    const int headSize = C / nHead;
    const float scale = 1.0f / sqrt((float)headSize);
    
    // 每个线程处理一个输出元素
    const int global_id = get_global_id(0);
    const int total_elements = B * T * C;
    
    if (global_id >= total_elements) return;
    
    // 分解全局ID
    const int batch = global_id / (T * C);
    const int seq_pos = (global_id % (T * C)) / C;
    const int channel = global_id % C;
    const int head = channel / headSize;
    const int head_channel = channel % headSize;
    
    // 1. 计算当前位置的Q值
    float q_val = 0.0f;
    for (int c_in = 0; c_in < C; c_in++) {
        int x_idx = (batch * T + seq_pos) * C + c_in;
        int wq_idx = c_in * (3 * C) + channel;
        q_val += x[x_idx] * c_attn[wq_idx];
    }
    
    // 2. 计算注意力分数（避免使用大数组）
    float max_score = SOFTMAX_MIN;
    
    // 第一遍：找最大值
    for (int j = 0; j <= seq_pos; j++) {
        float k_val = 0.0f;
        
        // 计算位置j的K值
        for (int c_in = 0; c_in < C; c_in++) {
            int x_idx = (batch * T + j) * C + c_in;
            int wk_idx = c_in * (3 * C) + C + channel;
            k_val += x[x_idx] * c_attn[wk_idx];
        }
        
        float score = q_val * k_val * scale;
        if (score > max_score) {
            max_score = score;
        }
    }
    
    // 第二遍：计算softmax分母
    float sum_exp = 0.0f;
    for (int j = 0; j <= seq_pos; j++) {
        float k_val = 0.0f;
        
        for (int c_in = 0; c_in < C; c_in++) {
            int x_idx = (batch * T + j) * C + c_in;
            int wk_idx = c_in * (3 * C) + C + channel;
            k_val += x[x_idx] * c_attn[wk_idx];
        }
        
        float score = q_val * k_val * scale;
        sum_exp += exp(score - max_score);
    }
    
    // 第三遍：计算加权和
    float attn_output = 0.0f;
    for (int j = 0; j <= seq_pos; j++) {
        float k_val = 0.0f;
        float v_val = 0.0f;
        
        for (int c_in = 0; c_in < C; c_in++) {
            int x_idx = (batch * T + j) * C + c_in;
            int wk_idx = c_in * (3 * C) + C + channel;
            int wv_idx = c_in * (3 * C) + 2 * C + channel;
            
            k_val += x[x_idx] * c_attn[wk_idx];
            v_val += x[x_idx] * c_attn[wv_idx];
        }
        
        float score = q_val * k_val * scale;
        float attention_weight = exp(score - max_score) / sum_exp;
        attn_output += attention_weight * v_val;
    }
    
    // 3. 输出投影
    float final_output = 0.0f;
    for (int c_in = 0; c_in < C; c_in++) {
        int proj_idx = c_in * C + channel;
        final_output += attn_output * c_proj[proj_idx];
    }
    
    output[global_id] = final_output;
}