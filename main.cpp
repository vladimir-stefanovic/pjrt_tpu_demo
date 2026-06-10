// ===--------------------------------------------------------------------=== //
// pjrt-demo: run a Mosaic IR kernel on TPU through PJRT
// ===--------------------------------------------------------------------=== //
//
// End-to-end flow:
//
//   Mosaic MLIR text file
//       |
//       v  WrapMosaicInStableHlo
//   StableHLO with @tpu_custom_call(... backend_config = <mosaic text>)
//       |
//       v  PJRT_Client_Compile
//   PJRT_LoadedExecutable
//       |
//       v  PJRT_Client_BufferFromHostBuffer (x2, for lhs/rhs)
//   PJRT_Buffer* inputs
//       |
//       v  PJRT_LoadedExecutable_Execute
//   PJRT_Buffer* output
//       |
//       v  PJRT_Buffer_ToHostBuffer
//   host float[128]
//
// The only compile-time dependency is xla/pjrt/c/pjrt_c_api.h (plain C).
// The runtime dependency is the PJRT-exporting TPU shared library
// (e.g. libtpu.so) whose path is passed on the command line.
// ===--------------------------------------------------------------------=== //

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

// ===--------------------------------------------------------------------=== //
// Error helper
// ===--------------------------------------------------------------------=== //
//
// Every PJRT call returns a PJRT_Error* (nullptr on success). On error we
// pull the message out via PJRT_Error_Message and destroy the error via
// PJRT_Error_Destroy.

bool CheckError(const PJRT_Api* api, PJRT_Error* error,
                const char* context) {
    if (error == nullptr) return true;

    PJRT_Error_Message_Args msg_args{};
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.error = error;
    api->PJRT_Error_Message(&msg_args);

    std::cerr << context << ": "
              << std::string(msg_args.message, msg_args.message_size)
              << "\n";

    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);
    return false;
}

// ===--------------------------------------------------------------------=== //
// File IO
// ===--------------------------------------------------------------------=== //

std::string LoadTextFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open " << path << "\n";
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// ===--------------------------------------------------------------------=== //
// StableHLO wrapping
// ===--------------------------------------------------------------------=== //

std::string EscapeMlirString(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() * 2);
    for (char c : raw) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

// Standard base64 encoder. jaxlib's `as_tpu_kernel` uses base64 for the
// JSON backend_config's "body" field. We replicate that format.
std::string Base64Encode(const std::string& data) {
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= data.size()) {
        uint32_t v = (static_cast<uint8_t>(data[i]) << 16) |
                     (static_cast<uint8_t>(data[i + 1]) << 8) |
                     static_cast<uint8_t>(data[i + 2]);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += tbl[v & 63];
        i += 3;
    }
    if (i < data.size()) {
        uint32_t v = static_cast<uint8_t>(data[i]) << 16;
        if (i + 1 < data.size())
            v |= static_cast<uint8_t>(data[i + 1]) << 8;
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += (i + 1 < data.size()) ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

// Build the StableHLO wrapper with the correct Mosaic backend_config format.
// The JSON shape comes from jaxlib's `CustomCallBackendConfig.to_json()`:
//     {"custom_call_config": {"body": "<base64>"}}
//
// The body is the Mosaic MLIR bytes (text is accepted by ir.Module.parse,
// though jaxlib normally feeds bytecode). We try text first; if libtpu
// rejects it we'd need a bytecode pre-processing step.
std::string WrapMosaicInStableHlo(const std::string& mosaic_text) {
    const std::string body_b64 = Base64Encode(mosaic_text);
    // Inner JSON; escape the embedded quotes for the MLIR string literal.
    const std::string json =
        std::string("{\\\"custom_call_config\\\": {\\\"body\\\": \\\"")
        + body_b64 + "\\\"}}";

    return
        "module @wrapper {\n"
        "  func.func @main(\n"
        "      %lhs: tensor<8x128xf32>,\n"
        "      %rhs: tensor<8x128xf32>\n"
        "  ) -> tensor<8x128xf32> {\n"
        "    %out = stablehlo.custom_call @tpu_custom_call(%lhs, %rhs) {\n"
        "      backend_config = \"" + json + "\",\n"
        "      api_version = 2 : i32\n"
        "    } : (tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>\n"
        "    return %out : tensor<8x128xf32>\n"
        "  }\n"
        "}\n";
}

// DIAGNOSTIC: a minimal StableHLO that just returns its first argument.
// Used to distinguish "compile path broken" from "Mosaic custom_call broken".
std::string BuildTrivialPassThroughStableHlo() {
    return
        "module @passthrough {\n"
        "  func.func @main(\n"
        "      %lhs: tensor<128xf32>,\n"
        "      %rhs: tensor<128xf32>\n"
        "  ) -> tensor<128xf32> {\n"
        "    return %lhs : tensor<128xf32>\n"
        "  }\n"
        "}\n";
}

// ===--------------------------------------------------------------------=== //
// PJRT plugin loading
// ===--------------------------------------------------------------------=== //

using GetPjrtApiFn = const PJRT_Api* (*)();

const PJRT_Api* LoadTpuPjrtApi(const char* plugin_path) {
    void* handle = dlopen(plugin_path, RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "dlopen(\"" << plugin_path << "\") failed: "
                  << dlerror() << "\n";
        return nullptr;
    }
    dlerror();
    auto get_api = reinterpret_cast<GetPjrtApiFn>(
        dlsym(handle, "GetPjrtApi"));
    const char* err = dlerror();
    if (err != nullptr || get_api == nullptr) {
        std::cerr << "dlsym(\"GetPjrtApi\") failed: "
                  << (err ? err : "symbol returned null") << "\n";
        dlclose(handle);
        return nullptr;
    }
    // Intentionally leak `handle`: the returned PJRT_Api points into the
    // loaded library, so unloading it would invalidate the function table.
    return get_api();
}

// ===--------------------------------------------------------------------=== //
// Wait on a PJRT_Event and destroy it.
// ===--------------------------------------------------------------------=== //

bool AwaitEventAndDestroy(const PJRT_Api* api, PJRT_Event* event,
                          const char* context) {
    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = event;
    if (!CheckError(api, api->PJRT_Event_Await(&await_args), context)) {
        return false;
    }

    PJRT_Event_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = event;
    api->PJRT_Event_Destroy(&destroy_args);
    return true;
}

}  // namespace

// ===--------------------------------------------------------------------=== //
// main
// ===--------------------------------------------------------------------=== //

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " <mosaic_mlir_file> <libtpu_so> [<output_blob_path>]\n\n"
        << "Arguments:\n"
        << "  <mosaic_mlir_file>   Mosaic MLIR source file (e.g. vector_add.mlir)\n"
        << "  <libtpu_so>          TPU PJRT plugin exporting GetPjrtApi\n"
        << "                       (e.g. .../site-packages/libtpu/libtpu.so)\n"
        << "  <output_blob_path>   Optional. If given, the compiled PJRT\n"
        << "                       executable is serialized to this path\n"
        << "                       after compilation.\n";
}

int main(int argc, char** argv) {
    // Scan for the optional --trivial flag (diagnostic only).
    bool trivial = false;
    bool hlo = false;
    std::vector<char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trivial") == 0) {
            trivial = true;
        } else if (std::strcmp(argv[i], "--hlo") == 0) {
            hlo = true;
	} else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 2 && positional.size() != 3) {
        PrintUsage(argc > 0 ? argv[0] : "pjrt-demo");
        return 1;
    }
    const char* mosaic_path = positional[0];
    const char* libtpu_path = positional[1];
    const char* blob_out_path =
        (positional.size() == 3) ? positional[2] : nullptr;

    std::cout << "=== pjrt-demo ===\n";
    std::cout << "Mosaic MLIR: " << mosaic_path << "\n";
    std::cout << "PJRT plugin: " << libtpu_path << "\n\n";

    // ---- 1. Load PJRT plugin ------------------------------------------
    const PJRT_Api* api = LoadTpuPjrtApi(libtpu_path);
    if (api == nullptr) return 1;
    std::cout << "[1] Loaded PJRT plugin\n";

    // ---- 2. Initialize plugin -----------------------------------------
    {
        PJRT_Plugin_Initialize_Args init_args{};
        init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Plugin_Initialize(&init_args),
                        "PJRT_Plugin_Initialize")) {
            return 1;
        }
    }
    std::cout << "[2] Initialized plugin\n";

    // ---- 3. Create PJRT client ----------------------------------------
    PJRT_Client* client = nullptr;
    {
        PJRT_Client_Create_Args create_args{};
        create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        if (!CheckError(api, api->PJRT_Client_Create(&create_args),
                        "PJRT_Client_Create")) {
            return 1;
        }
        client = create_args.client;
    }
    std::cout << "[3] Created PJRT client\n";

    // ---- 4. Get the first addressable device --------------------------
    PJRT_Device* device = nullptr;
    {
        PJRT_Client_AddressableDevices_Args dev_args{};
        dev_args.struct_size =
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        dev_args.client = client;
        if (!CheckError(api, api->PJRT_Client_AddressableDevices(&dev_args),
                        "PJRT_Client_AddressableDevices")) {
            return 1;
        }
        if (dev_args.num_addressable_devices == 0) {
            std::cerr << "No addressable devices.\n";
            return 1;
        }
        device = dev_args.addressable_devices[0];
        std::cout << "[4] Found " << dev_args.num_addressable_devices
                  << " addressable device(s); using device[0]\n";
    }

    // ---- 5. Load Mosaic MLIR and wrap in StableHLO --------------------
    std::string stablehlo;
    if (trivial) {
        stablehlo = BuildTrivialPassThroughStableHlo();
        std::cout << "[5] --trivial: using pass-through StableHLO ("
                  << stablehlo.size() << " bytes), ignoring "
                  << mosaic_path << "\n";
       } else if (hlo) {
           std::cout << "loading stable hlo;\n";
           stablehlo = LoadTextFile(mosaic_path);
           std::cout << "loaded stable hlo;\n";
    } else {
        std::string mosaic_text = LoadTextFile(mosaic_path);
        if (mosaic_text.empty()) return 1;
        stablehlo = WrapMosaicInStableHlo(mosaic_text);
        std::cout << "[5] Loaded Mosaic (" << mosaic_text.size()
                  << " bytes) and wrapped into StableHLO ("
                  << stablehlo.size() << " bytes)\n";
    }

    // ---- 6. Compile ---------------------------------------------------
    PJRT_LoadedExecutable* executable = nullptr;
    {
        static constexpr char kFormat[] = "mlir";
        PJRT_Program program{};
        program.struct_size = PJRT_Program_STRUCT_SIZE;
        program.code = const_cast<char*>(stablehlo.data());
        program.code_size = stablehlo.size();
        program.format = kFormat;
        program.format_size = sizeof(kFormat) - 1;  // 4, without NUL

        // Minimal CompileOptionsProto (from xla/pjrt/compile_options.proto):
        //   CompileOptionsProto.executable_build_options (field 3, message)
        //     ExecutableBuildOptionsProto.num_replicas   (field 4, int64) = 1
        //     ExecutableBuildOptionsProto.num_partitions (field 5, int64) = 1
        //
        // Wire format:
        //   0x1A 0x04  = tag(field=3, wire=length-delim), length=4
        //     0x20 0x01 = tag(field=4, wire=varint), value=1    (num_replicas)
        //     0x28 0x01 = tag(field=5, wire=varint), value=1    (num_partitions)
        //
        // Without this, PJRT rejects with "Invalid (replica_count,
        // computation_count) pair: (0,0)".
        static constexpr unsigned char kCompileOptions[] = {
            0x1A, 0x04, 0x20, 0x01, 0x28, 0x01,
        };

        PJRT_Client_Compile_Args compile_args{};
        compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
        compile_args.client = client;
        compile_args.program = &program;
        compile_args.compile_options =
            reinterpret_cast<const char*>(kCompileOptions);
        compile_args.compile_options_size = sizeof(kCompileOptions);
        if (!CheckError(api, api->PJRT_Client_Compile(&compile_args),
                        "PJRT_Client_Compile")) {
            return 1;
        }
        executable = compile_args.executable;
    }
    std::cout << "[6] Compiled StableHLO -> PJRT_LoadedExecutable\n";

    // ---- 6b. (optional) Serialize executable to disk ------------------
    if (blob_out_path != nullptr) {
        // Get the un-loaded Executable view of the LoadedExecutable.
        PJRT_LoadedExecutable_GetExecutable_Args get_args{};
        get_args.struct_size =
            PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
        get_args.loaded_executable = executable;
        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_GetExecutable(&get_args),
                        "PJRT_LoadedExecutable_GetExecutable")) {
            return 1;
        }
        PJRT_Executable* plain_exe = get_args.executable;

        // Serialize it.
        PJRT_Executable_Serialize_Args ser_args{};
        ser_args.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
        ser_args.executable = plain_exe;
        if (!CheckError(api, api->PJRT_Executable_Serialize(&ser_args),
                        "PJRT_Executable_Serialize")) {
            return 1;
        }

        // Write to disk.
        std::ofstream out(blob_out_path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open " << blob_out_path
                      << " for writing\n";
            return 1;
        }
        out.write(ser_args.serialized_bytes,
                  static_cast<std::streamsize>(
                      ser_args.serialized_bytes_size));
        out.close();
        std::cout << "[6b] Serialized executable ("
                  << ser_args.serialized_bytes_size << " bytes) to "
                  << blob_out_path << "\n";

        // Free the backing memory and the plain_exe wrapper.
        ser_args.serialized_executable_deleter(
            ser_args.serialized_executable);

        PJRT_Executable_Destroy_Args exe_destroy{};
        exe_destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
        exe_destroy.executable = plain_exe;
        api->PJRT_Executable_Destroy(&exe_destroy);
    }

    // ---- 7. Prepare host-side inputs ----------------------------------
    constexpr size_t ROWS = 8;
    constexpr size_t COLS = 128;
    constexpr size_t N = ROWS * COLS;  // 1024
    std::vector<float> host_lhs(N), host_rhs(N);
    for (size_t i = 0; i < N; ++i) {
        host_lhs[i] = static_cast<float>(i);
        host_rhs[i] = 1.0f;
    }

    // ---- 8. Upload inputs to device -----------------------------------
    auto upload = [&](const std::vector<float>& host_data)
        -> PJRT_Buffer* {
        int64_t dims[] = {static_cast<int64_t>(ROWS),
                          static_cast<int64_t>(COLS)};

        PJRT_Client_BufferFromHostBuffer_Args args{};
        args.struct_size =
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        args.client = client;
        args.data = host_data.data();
        args.type = PJRT_Buffer_Type_F32;
        args.dims = dims;
        args.num_dims = 2;
        args.byte_strides = nullptr;
        args.num_byte_strides = 0;
        args.host_buffer_semantics =
            PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        args.device = device;
        args.memory = nullptr;
        args.device_layout = nullptr;

        if (!CheckError(api, api->PJRT_Client_BufferFromHostBuffer(&args),
                        "PJRT_Client_BufferFromHostBuffer")) {
            return nullptr;
        }
        // Wait for the H->D transfer to complete before freeing host_data
        // (which happens at end of scope).
        if (!AwaitEventAndDestroy(api, args.done_with_host_buffer,
                                  "done_with_host_buffer await")) {
            return nullptr;
        }
        return args.buffer;
    };

    PJRT_Buffer* buf_lhs = upload(host_lhs);
    PJRT_Buffer* buf_rhs = upload(host_rhs);
    if (buf_lhs == nullptr || buf_rhs == nullptr) return 1;
    std::cout << "[7] Uploaded 2 x " << N << " floats to device\n";

    // ---- 9. Execute ---------------------------------------------------
    PJRT_Buffer* output_buffers[1] = {nullptr};
    PJRT_Buffer** output_list = output_buffers;
    PJRT_Buffer* input_args[2] = {buf_lhs, buf_rhs};
    // Non-const to match PJRT 0.23 (PJRT_Buffer***); implicit const-adding
    // conversion covers the newer type PJRT_Buffer* const* const*.
    PJRT_Buffer** input_list[1] = {input_args};
    {
        PJRT_ExecuteOptions exec_options{};
        exec_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

        PJRT_LoadedExecutable_Execute_Args exec_args{};
        exec_args.struct_size =
            PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        exec_args.executable = executable;
        exec_args.options = &exec_options;
        exec_args.argument_lists = input_list;
        exec_args.num_devices = 1;
        exec_args.num_args = 2;
        exec_args.output_lists = &output_list;
        exec_args.device_complete_events = nullptr;
        exec_args.execute_device = nullptr;

        if (!CheckError(api,
                        api->PJRT_LoadedExecutable_Execute(&exec_args),
                        "PJRT_LoadedExecutable_Execute")) {
            return 1;
        }
    }
    std::cout << "[8] Executed kernel\n";

    // ---- 10. Download output ------------------------------------------
    std::vector<float> host_out(N);
    {
        PJRT_Buffer_ToHostBuffer_Args args{};
        args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        args.src = output_buffers[0];
        args.host_layout = nullptr;
        args.dst = host_out.data();
        args.dst_size = host_out.size() * sizeof(float);
        if (!CheckError(api, api->PJRT_Buffer_ToHostBuffer(&args),
                        "PJRT_Buffer_ToHostBuffer")) {
            return 1;
        }
        if (!AwaitEventAndDestroy(api, args.event,
                                  "PJRT_Buffer_ToHostBuffer await")) {
            return 1;
        }
    }
    std::cout << "[9] Downloaded output\n\n";

    // ---- 11. Show results ---------------------------------------------
    std::cout << "Result (first 8 elements):\n  ";
    for (int i = 0; i < 8; ++i) {
        std::cout << host_out[i] << " ";
    }
    std::cout << "\nExpected:\n  ";
    for (int i = 0; i < 8; ++i) {
        std::cout << (host_lhs[i] + host_rhs[i]) << " ";
    }
    std::cout << "\n";

    // ---- 12. Cleanup --------------------------------------------------
    auto destroy_buffer = [&](PJRT_Buffer* buf) {
        if (buf == nullptr) return;
        PJRT_Buffer_Destroy_Args a{};
        a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        a.buffer = buf;
        api->PJRT_Buffer_Destroy(&a);
    };
    destroy_buffer(buf_lhs);
    destroy_buffer(buf_rhs);
    destroy_buffer(output_buffers[0]);

    {
        PJRT_LoadedExecutable_Destroy_Args a{};
        a.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        a.executable = executable;
        api->PJRT_LoadedExecutable_Destroy(&a);
    }
    {
        PJRT_Client_Destroy_Args a{};
        a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        a.client = client;
        api->PJRT_Client_Destroy(&a);
    }

    std::cout << "\n[done]\n";
    return 0;
}
