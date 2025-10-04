# GGML CPU+GPU Hybrid Execution Architecture Analysis

**Date:** 2025-10-03
**Target:** whisper.cpp with Metal backend (macOS)
**Goal:** Analyze feasibility of hybrid CPU+GPU execution (3:1 GPU:CPU ratio for audio chunks)

---

## Executive Summary

**Key Findings:**
1. ✅ GGML **DOES** support multiple backends simultaneously via `ggml_backend_sched`
2. ✅ Metal backend does **NOT** monopolize all operations - scheduler decides per-operation
3. ⚠️ **Current Limitation:** whisper.cpp processes audio chunks sequentially, not in parallel
4. 🔧 **Implementation Level:** Requires **whisper.cpp** level modifications, not GGML core changes
5. 📊 **Complexity:** Medium - requires chunk-level backend assignment and pipeline management

**Conclusion:** Hybrid execution is architecturally possible but requires application-level changes to whisper.cpp's audio processing pipeline.

---

## 1. GGML Backend Architecture

### 1.1 Backend System Overview

GGML implements a sophisticated multi-backend scheduler that enables **heterogeneous execution** across different hardware devices:

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:1602-1658`

```cpp
ggml_backend_sched_t ggml_backend_sched_new(
    ggml_backend_t * backends,           // Array of backend instances
    ggml_backend_buffer_type_t * bufts,
    int n_backends,                       // Number of backends
    size_t graph_size,
    bool parallel,                        // Pipeline parallelism
    bool op_offload)                      // Offload expensive ops to GPU
```

**Key Architecture Points:**

1. **Multiple Backends Support:** The scheduler explicitly supports `n_backends` (up to `GGML_SCHED_MAX_BACKENDS`)
2. **Backend Priority:** Lower index = higher priority (line 734)
3. **Backend Device Types:**
   - `GGML_BACKEND_DEVICE_TYPE_CPU`
   - `GGML_BACKEND_DEVICE_TYPE_GPU` (Metal, CUDA, Vulkan)
   - `GGML_BACKEND_DEVICE_TYPE_IGPU` (integrated GPU)
   - `GGML_BACKEND_DEVICE_TYPE_ACCEL` (accelerators)

**File:** `/Users/aleksander/Documents/projects3/ggml/include/ggml-backend.h:130-139`

### 1.2 Backend Selection Algorithm

The scheduler uses a **multi-pass algorithm** to assign operations to backends:

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:912-1050`

**Pass 1:** Assign operations with pre-allocated inputs (lines 931-966)
```cpp
// Operations with weights prefer the backend where weights reside
for (int i = 0; i < GGML_MAX_SRC; i++) {
    if (src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
        // Check if higher-priority backend wants to offload
        if (sched->op_offload && ggml_backend_offload_op(backend, tensor)) {
            return backend_id;  // Use GPU for expensive ops
        }
        return src_backend_id;  // Use backend where weights are
    }
}
```

**Pass 2:** Expand GPU backend assignments (lines 968-1046)
```cpp
// Expand GPU down
for (int i = 0; i < graph->n_nodes; i++) {
    if (cur_backend_id == sched->n_backends - 1) {
        // Skip CPU (lowest priority)
        cur_backend_id = -1;
    } else if (ggml_backend_supports_op(backend, node)) {
        *node_backend_id = cur_backend_id;  // Assign to GPU
    }
}
```

**Pass 3:** Upgrade to higher-priority backends if compatible (lines 1048+)

### 1.3 Graph Splitting Mechanism

After backend assignment, the scheduler creates **splits** - subgraphs executed on single backends:

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:668-676`

```cpp
struct ggml_backend_sched_split {
    int backend_id;                          // Which backend executes this split
    int i_start;                             // Start node index
    int i_end;                               // End node index
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];  // Inputs from other splits
    int n_inputs;                            // Number of cross-backend inputs
    struct ggml_cgraph graph;                // Subgraph for this split
};
```

**Cross-Backend Data Transfer:**
- When split N+1 needs output from split N on different backend
- Automatic tensor copy between backends (line 1180-1250)
- Uses `ggml_backend_tensor_copy_async()` for efficiency

---

## 2. Metal Backend Integration

### 2.1 Metal Backend Structure

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-metal/ggml-metal-device.cpp`

Metal backend is implemented as a **standard GGML backend** - it does NOT monopolize operations:

```cpp
ggml_backend_t ggml_backend_metal_init(void) {
    // Initialize Metal device
    // Register as backend in scheduler
    // Implements ggml_backend_i interface
}
```

**Key Metal Files:**
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-metal/ggml-metal-context.m` - Context management
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-metal/ggml-metal-device.m` - Device initialization
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-metal/ggml-metal-ops.cpp` - Operation implementations

### 2.2 Metal Backend Registration

**File:** `/Users/aleksander/Documents/projects3/ggml/include/ggml-metal.h:42-58`

```cpp
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);
GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);
```

Metal backend **participates in scheduler** like any other backend - no special monopoly.

### 2.3 Operation Support Query

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend-impl.h:169-177`

```cpp
struct ggml_backend_device_i {
    // Check if backend can compute operation
    bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

    // Check if backend wants to run expensive operation
    // (even if weights are in incompatible buffer)
    bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);
};
```

Metal implements these to declare which operations it can handle efficiently.

---

## 3. Whisper.cpp Backend Integration

### 3.1 Scheduler Wrapper

**File:** `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp:531-567`

```cpp
struct whisper_sched {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

static bool whisper_sched_graph_init(
    struct whisper_sched & allocr,
    std::vector<ggml_backend_t> backends,
    std::function<struct ggml_cgraph *()> && get_graph)
{
    // Create scheduler with ALL provided backends
    sched = ggml_backend_sched_new(
        backends.data(),          // Multiple backends possible
        nullptr,
        backends.size(),
        WHISPER_MAX_NODES,
        false,                    // parallel = false (no pipeline parallelism)
        true);                    // op_offload = true (offload expensive ops to GPU)

    // Allocate graph on scheduler
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        return false;
    }

    return true;
}
```

**CRITICAL:** Line 552 shows `parallel = false` - whisper.cpp does NOT use pipeline parallelism currently.

### 3.2 Backend Initialization

Whisper.cpp creates separate schedulers for different model components:

**File:** `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp:3812-3815`

```cpp
ggml_backend_sched_free(state->sched_conv.sched);     // Conv1D encoder
ggml_backend_sched_free(state->sched_encode.sched);   // Audio encoder
ggml_backend_sched_free(state->sched_cross.sched);    // Cross-attention
ggml_backend_sched_free(state->sched_decode.sched);   // Text decoder
```

Each scheduler can theoretically use different backend combinations.

### 3.3 Current Backend Usage

**File:** `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp:162-182`

```cpp
static bool ggml_graph_compute_helper(
    struct ggml_cgraph * graph,
    int n_threads,
    ggml_abort_callback abort_callback,
    void * abort_callback_data)
{
    // Initialize SINGLE backend (CPU)
    ggml_backend_ptr backend {
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)
    };

    return ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS;
}
```

**Observation:** This helper uses ONLY CPU backend. Multi-backend version exists at line 184-206.

### 3.4 Audio Chunk Processing

**Critical Finding:** Whisper.cpp processes audio chunks **sequentially**, not in parallel batches.

The encoder processes each 30-second chunk one at a time, so there's no opportunity to distribute chunks across backends without architectural changes.

---

## 4. Hybrid Execution Feasibility Analysis

### 4.1 Current Architecture Assessment

**What Works:**
- ✅ GGML scheduler supports multiple backends simultaneously
- ✅ Backend selection is per-operation, not global
- ✅ Metal backend integrates cleanly without monopolizing
- ✅ Cross-backend tensor copies work automatically

**What Doesn't Work:**
- ❌ Whisper.cpp doesn't process chunks in parallel
- ❌ No API to assign specific chunks to specific backends
- ❌ Sequential processing prevents load distribution

### 4.2 Implementation Options

#### Option A: Modify GGML Scheduler (NOT RECOMMENDED)

**Difficulty:** High
**Location:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:912-1050`

**Approach:** Add custom backend selection logic to scheduler
```cpp
// In ggml_backend_sched_backend_id_from_cur()
if (is_whisper_chunk && chunk_index % 4 == 3) {
    return cpu_backend_id;  // Every 4th chunk on CPU
} else {
    return gpu_backend_id;  // Other chunks on GPU
}
```

**Why NOT Recommended:**
- Breaks abstraction - scheduler shouldn't know about application logic
- Whisper-specific hacks in generic library
- Doesn't solve sequential chunk processing issue

#### Option B: Per-Operation Backend Assignment (PARTIAL SOLUTION)

**Difficulty:** Medium
**Location:** whisper.cpp application level

**Approach:** Use `ggml_backend_sched_set_tensor_backend()`

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:1812-1819`

```cpp
void ggml_backend_sched_set_tensor_backend(
    ggml_backend_sched_t sched,
    struct ggml_tensor * node,
    ggml_backend_t backend)
{
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    tensor_backend_id(node) = backend_index;
}
```

**Implementation in whisper.cpp:**
```cpp
// In encoder graph construction
static struct ggml_cgraph * whisper_build_graph_encoder(
    whisper_context & wctx,
    whisper_state & wstate,
    int chunk_index)  // NEW PARAMETER
{
    // Build encoder graph
    struct ggml_cgraph * gf = ...;

    // Assign backend based on chunk index
    ggml_backend_t target_backend;
    if (chunk_index % 4 == 3) {
        target_backend = wctx.backend_cpu;
    } else {
        target_backend = wctx.backend_metal;
    }

    // Assign ALL nodes in graph to selected backend
    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_backend_sched_set_tensor_backend(
            wstate.sched_encode.sched,
            gf->nodes[i],
            target_backend);
    }

    return gf;
}
```

**Limitations:**
- Still processes chunks sequentially
- No actual parallel execution
- Questionable benefit (context switching overhead)

#### Option C: Parallel Chunk Pipeline (RECOMMENDED)

**Difficulty:** High
**Location:** whisper.cpp audio processing pipeline

**Approach:** Process multiple chunks concurrently with different backends

**Architecture:**

```cpp
struct whisper_chunk_context {
    int chunk_index;
    ggml_backend_t backend;
    ggml_backend_sched_t sched;
    whisper_state state;
    std::vector<float> mel_data;
};

static void whisper_encode_parallel(
    whisper_context & wctx,
    std::vector<float> & audio_samples)
{
    // Split audio into chunks
    const int n_chunks = audio_samples.size() / (WHISPER_SAMPLE_RATE * 30);

    // Create chunk contexts
    std::vector<whisper_chunk_context> chunks(4);

    for (int i = 0; i < 4 && i < n_chunks; i++) {
        chunks[i].chunk_index = i;

        // Assign backend: 3 GPU, 1 CPU
        if (i == 3) {
            chunks[i].backend = wctx.backend_cpu;
        } else {
            chunks[i].backend = wctx.backend_metal;
        }

        // Create separate scheduler for this chunk
        chunks[i].sched = ggml_backend_sched_new(
            &chunks[i].backend,
            nullptr,
            1,  // Single backend per chunk
            WHISPER_MAX_NODES,
            false,
            true);

        // Extract mel spectrogram for this chunk
        chunks[i].mel_data = extract_mel_chunk(audio_samples, i);
    }

    // Process chunks in parallel
    std::vector<std::thread> workers;
    for (auto & chunk : chunks) {
        workers.emplace_back([&chunk, &wctx]() {
            // Build encoder graph for this chunk
            auto * gf = whisper_build_graph_encoder(
                wctx,
                chunk.state,
                chunk.chunk_index);

            // Execute on assigned backend
            ggml_backend_sched_graph_compute(chunk.sched, gf);
        });
    }

    // Wait for all chunks
    for (auto & worker : workers) {
        worker.join();
    }

    // Merge results
    merge_chunk_results(chunks, wctx);
}
```

**Advantages:**
- True parallel execution
- Efficient GPU/CPU utilization
- Natural load distribution (3:1 ratio)

**Challenges:**
- Requires refactoring whisper.cpp encoder pipeline
- Memory management complexity (4x state overhead)
- Synchronization and result merging
- Potential accuracy issues (cross-chunk context)

#### Option D: Pipeline Parallelism (ALTERNATIVE)

**Difficulty:** Very High
**Location:** GGML scheduler + whisper.cpp

**Approach:** Use GGML's built-in pipeline parallelism

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:1618`

```cpp
sched = ggml_backend_sched_new(
    backends,
    nullptr,
    n_backends,
    WHISPER_MAX_NODES,
    true,  // parallel = TRUE (enable pipeline parallelism)
    true);
```

**How it works:**
- GGML creates `GGML_SCHED_MAX_COPIES` copies of graph
- Different copies execute on different backends
- Synchronized via events (line 711)

**Limitations:**
- Designed for layer-level parallelism (e.g., different Transformer layers)
- Not designed for chunk-level parallelism
- Would require extensive whisper.cpp modifications

---

## 5. Implementation Roadmap

### 5.1 Recommended Approach: Option C (Parallel Chunk Pipeline)

**Phase 1: Proof of Concept (1-2 weeks)**

1. **Refactor Encoder State Management**
   - File: `whisper.cpp/src/whisper.cpp` (whisper_state structure)
   - Make `whisper_state` thread-safe
   - Support multiple concurrent encoder states

2. **Implement Chunk-Level Backend Assignment**
   - Create `whisper_chunk_context` structure
   - Initialize separate schedulers per chunk
   - Assign backends (3 GPU, 1 CPU)

3. **Parallel Execution Framework**
   - Thread pool for chunk processing
   - Synchronization primitives
   - Result merging logic

**Phase 2: Integration (2-3 weeks)**

4. **Mel Spectrogram Extraction**
   - Parallelize mel computation
   - Separate mel buffers per chunk

5. **Cross-Chunk Context Handling**
   - Investigate impact on accuracy
   - Implement context overlap if needed
   - Handle chunk boundaries

6. **Memory Management**
   - Allocate/deallocate chunk contexts
   - Manage buffer lifetimes
   - Prevent memory leaks

**Phase 3: Optimization (1-2 weeks)**

7. **Performance Tuning**
   - Minimize context switching
   - Optimize tensor copies
   - Reduce synchronization overhead

8. **Load Balancing**
   - Dynamic backend selection based on chunk complexity
   - Adaptive GPU/CPU ratio

9. **Testing & Validation**
   - Accuracy comparison with sequential processing
   - Performance benchmarks
   - Edge case handling

### 5.2 Code Modifications Summary

**Files to Modify:**

1. `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp`
   - Lines 531-567: Extend `whisper_sched` for multi-backend
   - Lines ~2000-2500: Refactor encoder pipeline
   - Add new functions: `whisper_encode_parallel()`, `whisper_chunk_context_init()`

2. `/Users/aleksander/Documents/projects3/whisper.cpp/include/whisper.h`
   - Add new API for parallel chunk processing
   - Expose backend selection controls

3. **No GGML Core Modifications Required**
   - GGML scheduler already supports everything needed
   - Use existing APIs: `ggml_backend_sched_new()`, `ggml_backend_sched_graph_compute()`

### 5.3 Complexity Estimate

**Lines of Code Changed:** ~1000-1500 LOC
**Development Time:** 4-7 weeks
**Risk Level:** Medium

**Major Risks:**
1. Accuracy degradation due to chunk independence
2. Memory overhead (4x state duplication)
3. Synchronization bugs
4. Race conditions in result merging

---

## 6. Alternative Approaches

### 6.1 Weighted Round-Robin (Simpler)

Instead of fixed 3:1 ratio, use weighted scheduling:

```cpp
int select_backend_for_chunk(int chunk_index) {
    // GPU gets 75% chunks, CPU gets 25%
    static int gpu_chunks = 0, cpu_chunks = 0;

    float gpu_ratio = (float)gpu_chunks / (gpu_chunks + cpu_chunks + 1);

    if (gpu_ratio < 0.75) {
        gpu_chunks++;
        return BACKEND_METAL;
    } else {
        cpu_chunks++;
        return BACKEND_CPU;
    }
}
```

**Pros:** Simpler implementation
**Cons:** Still sequential, no real parallelism

### 6.2 Operation-Level Splitting (Not Recommended)

Split within a chunk - e.g., attention on GPU, FFN on CPU.

**Why Not:**
- Excessive tensor copies
- Poor cache locality
- Minimal benefit
- Complex debugging

---

## 7. Environment Variable Debugging

### 7.1 GGML_SCHED_DEBUG

**File:** `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:1615-1616`

```cpp
const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
```

**Usage:**
```bash
export GGML_SCHED_DEBUG=2
./main -m model.bin audio.wav
```

**Output:** Shows which backend executes each operation (line 843-880)

**Reference:** [llama.cpp Discussion #8146](https://github.com/ggml-org/llama.cpp/discussions/8146)

> "Some layers are run on the CPU and others on the GPU, sequentially. You can set the GGML_SCHED_DEBUG environment variable to see what operations are being run on each device."

---

## 8. Related Research

### 8.1 GitHub Discussions

**Key Finding:** [whisper.cpp Discussion #1570](https://github.com/ggml-org/whisper.cpp/discussions/1570)

> **User:** "Using both the GPU and CPU simultaneously"
> **Maintainer:** "It could become feasible in the future, once the scheduler is fully implemented in the ggml backend."

**Analysis:** As of 2025, the scheduler IS fully implemented in GGML. The limitation is in whisper.cpp's sequential chunk processing.

### 8.2 Existing Multi-Backend Examples

**llama.cpp Implementation:**

llama.cpp DOES use multi-backend execution for layer offloading (`-ngl` parameter):

```bash
llama-cli -m model.gguf -ngl 32  # Offload 32 layers to GPU, rest on CPU
```

**Implementation:** Each transformer layer assigned to different backend based on `-ngl`.

**Difference from Whisper:**
- Transformer layers are independent (can run on different devices)
- Whisper chunks have temporal dependencies (harder to parallelize)

---

## 9. Conclusions & Recommendations

### 9.1 Answers to Original Questions

**Q1: Can GGML delegate tasks simultaneously to GPU and CPU?**
**A:** ✅ YES. The `ggml_backend_sched` explicitly supports multiple backends concurrently. Backend selection is per-operation, not global.

**Q2: Does Metal build "push everything to GPU" without distinction?**
**A:** ❌ NO. Metal is a standard backend registered in the scheduler. The scheduler decides which operations go to Metal based on:
   - Operation support (`supports_op()`)
   - Weight location (prefer backend where weights reside)
   - Offload preference (`offload_op()` for expensive operations)
   - User assignment (`ggml_backend_sched_set_tensor_backend()`)

**Q3: Do you need to edit whisper.cpp or GGML (scheduler/DAG)?**
**A:** **whisper.cpp ONLY.** GGML scheduler already provides all necessary functionality. Modifications needed:
   - Parallel chunk processing pipeline
   - Multiple concurrent encoder states
   - Backend assignment per chunk
   - Result merging logic

**Q4: How does backend selection work in GGML?**
**A:** Three-pass algorithm:
   1. **Pass 1:** Assign based on weight location and offload hints
   2. **Pass 2:** Expand GPU backend to adjacent compatible operations
   3. **Pass 3:** Upgrade to higher-priority backends if beneficial

   Result: Graph split into subgraphs (splits), each executed on single backend, with automatic cross-backend tensor copies.

### 9.2 Final Recommendation

**For 3:1 GPU:CPU chunk distribution:**

1. **Implement Option C (Parallel Chunk Pipeline)**
   - Best performance potential
   - Clean architecture
   - Aligns with GGML design

2. **Start with Proof of Concept**
   - 2 chunks only (1 GPU, 1 CPU)
   - Validate accuracy preservation
   - Measure actual speedup

3. **Validate Assumptions**
   - Does Metal performance justify complexity?
   - Is CPU truly idle during GPU processing?
   - What's the tensor copy overhead?

4. **Consider Simpler Alternative First**
   - Try existing pipeline parallelism (`parallel = true` in scheduler)
   - May already provide benefits without custom code

**Expected Outcome:**
- **Speedup:** 1.2-1.5x (if GPU is bottleneck)
- **Memory:** +300% overhead (4x state duplication)
- **Complexity:** Moderate to High

**Not Worth It If:**
- GPU is already saturated
- CPU is idle waiting for GPU
- Memory is constrained

---

## 10. Code References

### 10.1 Key Files

**GGML Core:**
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend.cpp:668-1828` - Scheduler implementation
- `/Users/aleksander/Documents/projects3/ggml/include/ggml-backend.h:24-370` - Backend API
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-backend-impl.h:1-258` - Backend interface

**Metal Backend:**
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-metal/ggml-metal-device.cpp` - Device management
- `/Users/aleksander/Documents/projects3/ggml/src/ggml-metal/ggml-metal-context.m` - Execution context
- `/Users/aleksander/Documents/projects3/ggml/include/ggml-metal.h:22-61` - Public API

**Whisper.cpp:**
- `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp:531-567` - Scheduler wrapper
- `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp:162-206` - Graph compute helpers
- `/Users/aleksander/Documents/projects3/whisper.cpp/src/whisper.cpp:3812-3815` - Scheduler cleanup

### 10.2 Important Line Numbers

**Scheduler Creation:**
- `ggml-backend.cpp:1602` - `ggml_backend_sched_new()`
- `ggml-backend.cpp:1611` - CPU backend requirement check
- `ggml-backend.cpp:1640-1650` - Backend registration loop

**Backend Selection:**
- `ggml-backend.cpp:734-741` - Backend priority lookup
- `ggml-backend.cpp:743-763` - Buffer-based backend selection
- `ggml-backend.cpp:776-831` - Full backend assignment logic
- `ggml-backend.cpp:814-827` - Weight-based preference + offload

**Graph Splitting:**
- `ggml-backend.cpp:912-1050` - Multi-pass splitting algorithm
- `ggml-backend.cpp:1144-1275` - Split creation and input management

**Manual Assignment API:**
- `ggml-backend.cpp:1812-1819` - `ggml_backend_sched_set_tensor_backend()`
- `ggml-backend.cpp:1821-1828` - `ggml_backend_sched_get_tensor_backend()`

---

## 11. Next Steps

**Immediate Actions:**

1. ✅ **Run with GGML_SCHED_DEBUG=2** to see current backend usage:
   ```bash
   export GGML_SCHED_DEBUG=2
   ./main -m ggml-large-v3.bin audio.wav
   ```
   Expected: Most operations on Metal, some on CPU (input/output)

2. ✅ **Benchmark Sequential Metal-only vs CPU-only:**
   - Measure single-chunk processing time on each backend
   - Determine if hybrid makes sense

3. ✅ **Prototype Simple Backend Assignment:**
   - Modify `whisper_build_graph_encoder()` to manually assign backends
   - Test if API works as expected

4. 🔄 **Evaluate Parallel Chunk Processing:**
   - If benchmarks show potential, implement PoC
   - Start with 2 concurrent chunks
   - Measure accuracy delta

**Long-term:**

5. 🔄 **Contribute to whisper.cpp:**
   - If successful, upstream parallel chunk support
   - Make backend selection configurable
   - Document performance characteristics

6. 🔄 **Explore GGML Pipeline Parallelism:**
   - Investigate `parallel = true` mode
   - May provide benefits without custom code

---

**Analysis Complete.**
