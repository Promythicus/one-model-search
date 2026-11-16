# Building the tools

`probe` (needs llama.cpp built at `~/models/llama.cpp/src/build`, commit ded522f):

    L=~/models/llama.cpp/src
    g++ -O2 -std=c++17 probe.cpp -o probe -I$L/include -I$L/common -I$L/ggml/include -I$L/vendor -I$L/src \
        -L$L/build/bin -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -lpthread

Extra llama.cpp targets built for this work: `cmake --build . --target llama-cvector-generator llama-quantize
llama-eval-callback llama-export-lora llama-tokenize`.

Checked 2026-11-16 on Qwen3-4B-Instruct Q4_K_M (CPU): `l_out` [36 layers, 2560], `ffn_swiglu` [36, 9728];
last-token and span-mean modes both write.
