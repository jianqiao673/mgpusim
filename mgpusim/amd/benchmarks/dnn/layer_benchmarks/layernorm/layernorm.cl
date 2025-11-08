__kernel void LayerNormForward(
    __global const float* input,
    __global float* output,
    const uint count,
    const uint hidden_size
) {
    int gid = get_global_id(0);
    int row = gid / hidden_size;
    int col = gid % hidden_size;

    // compute mean
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        sum += input[row * hidden_size + i];
    }
    float mean = sum / hidden_size;

    // compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float diff = input[row * hidden_size + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / hidden_size;

    // normalize
    float eps = 1e-5f;
    output[gid] = (input[gid] - mean) / sqrt(variance + eps);
}
