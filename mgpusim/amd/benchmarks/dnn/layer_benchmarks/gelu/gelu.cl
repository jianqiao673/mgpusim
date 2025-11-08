// gelu.cl
__kernel void GELUForward(const int count,
                          __global float* in,
                          __global float* out) {
    int index = get_global_id(0);
    if (index < count) {
        float x = in[index];
        float c = 0.79788456f;  // sqrt(2/pi)
        float x3 = x*x*x;
        float approx = 0.5f * x * (1 + c*(x + 0.044715f*x3));

        out[index] = approx;
    }
}
