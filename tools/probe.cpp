// probe: run texts through a GGUF model with llama.cpp and save chosen internal tensors.
// The microscope for the quantized file we actually run.
//
//   PROBE_PROMPTS=in.jsonl PROBE_OUT=dir PROBE_NAMES=l_out,ffn_swiglu ./probe -m model.gguf -ngl 99 -c 8192 -b 8192
//
// in.jsonl: one object per line, {"text": "..."}; if the text contains the marker U+241E once, the saved vector
// is the mean over the tokens after the marker, otherwise it is the last token's.
// Names (llama.cpp graph names for this architecture):  l_out = the stream after a layer;
// ffn_swiglu = the MLP neurons' activations (input of ffn_down);  ffn_out = what the MLP adds to the stream.
// PROBE_LAYERS=40,54 keeps only those layers (default: all).
// Output: dir/<name>.f32  float32 [texts, layers, width]  and dir/meta.json.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include <nlohmann/json.hpp>
#include <clocale>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct acc { std::vector<double> sum; std::vector<float> last; long seen = 0; long used = 0; };
struct state {
    std::vector<std::string> names; std::vector<int> layers; int from = -1;           // from: first token index of the span, -1 = last token only
    std::map<std::string, std::map<int, acc>> cur;           // name -> layer -> accumulator, for the text being run
    std::vector<float> buf;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = (state *) ud; const char * dash = strrchr(t->name, '-');
    if (!dash) return ask ? false : true;
    std::string name(t->name, dash - t->name); bool want = false;
    for (auto & n : s->names) want |= (n == name);
    if (ask) return want;
    if (!want || t->type != GGML_TYPE_F32) return true;
    int layer = atoi(dash + 1); long d = t->ne[0], T = t->ne[1];
    if (!s->layers.empty()) { bool ok = false; for (int l : s->layers) ok |= (l == layer); if (!ok) return true; }
    s->buf.resize(d * T); ggml_backend_tensor_get(t, s->buf.data(), 0, d * T * sizeof(float));
    acc & a = s->cur[name][layer];
    if (a.sum.empty()) { a.sum.assign(d, 0.0); a.last.assign(d, 0.f); }
    for (long j = 0; j < T; j++) {
        if (s->from >= 0 && a.seen + j >= s->from) { for (long i = 0; i < d; i++) a.sum[i] += s->buf[j * d + i]; a.used++; }
    }
    memcpy(a.last.data(), s->buf.data() + (T - 1) * d, d * sizeof(float)); a.seen += T;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * pf = getenv("PROBE_PROMPTS"); const char * od = getenv("PROBE_OUT"); const char * nm = getenv("PROBE_NAMES");
    if (!pf || !od) { fprintf(stderr, "set PROBE_PROMPTS and PROBE_OUT\n"); return 1; }
    state st; { std::string n = nm ? nm : "l_out"; size_t p; while ((p = n.find(',')) != std::string::npos) { st.names.push_back(n.substr(0, p)); n.erase(0, p + 1); } st.names.push_back(n); }
    if (const char * ls = getenv("PROBE_LAYERS")) { std::string n = ls; size_t p; while ((p = n.find(',')) != std::string::npos) { st.layers.push_back(atoi(n.substr(0, p).c_str())); n.erase(0, p + 1); } st.layers.push_back(atoi(n.c_str())); }
    common_params params; common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    params.cb_eval = cb; params.cb_eval_user_data = &st; params.warmup = false;
    auto init = common_init_from_params(params); auto * model = init->model(); auto * ctx = init->context();
    if (!model || !ctx) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model); const std::string mark = "\xE2\x90\x9E";
    std::ifstream in(pf); std::string line; std::map<std::string, FILE *> out; nlohmann::json meta; meta["texts"] = nlohmann::json::array(); long n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::string text = nlohmann::json::parse(line)["text"].get<std::string>(); std::vector<llama_token> toks; size_t m = text.find(mark);
        if (m == std::string::npos) { toks = common_tokenize(ctx, text, llama_vocab_get_add_bos(vocab), true); st.from = -1; }
        else {
            toks = common_tokenize(ctx, text.substr(0, m), llama_vocab_get_add_bos(vocab), true); st.from = (int) toks.size();
            auto tail = common_tokenize(ctx, text.substr(m + mark.size()), false, true); toks.insert(toks.end(), tail.begin(), tail.end());
        }
        st.cur.clear(); llama_memory_clear(llama_get_memory(ctx), true);
        common_batch batch = common_batch_get_one(ctx, toks);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { fprintf(stderr, "failed on text %ld\n", n); return 1; }
        for (auto & name : st.names) {
            auto & layers = st.cur[name];
            if (!out.count(name)) { out[name] = fopen((std::string(od) + "/" + name + ".f32").c_str(), "wb"); meta["shape"][name] = { (int) layers.size(), (int) layers.begin()->second.last.size() }; }
            for (auto & kv : layers) {
                acc & a = kv.second; std::vector<float> v(a.last);
                if (st.from >= 0 && a.used > 0) for (size_t i = 0; i < v.size(); i++) v[i] = (float) (a.sum[i] / a.used);
                fwrite(v.data(), sizeof(float), v.size(), out[name]);
            }
        }
        meta["texts"].push_back({ {"tokens", (int) toks.size()}, {"span_from", st.from} }); n++;
        if (n % 20 == 0) fprintf(stderr, "probe: %ld texts\n", n);
    }
    for (auto & kv : out) fclose(kv.second);
    std::ofstream(std::string(od) + "/meta.json") << meta.dump();
    fprintf(stderr, "probe: done, %ld texts\n", n); llama_backend_free(); return 0;
}
