// The v2 inspector checks the actual generated files and embedded plugin bytes on CPU.
#include <tensoracle/cubin.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace {
void require(bool value, char const* why) { if (!value) throw std::runtime_error(why); }
void rejected(fs::path const& root, tensoracle_status expected)
{
    try { tensoracle::inspect_bundle(root.c_str()); }
    catch (tensoracle::error const& error) {
        if (error.status() != expected) throw std::runtime_error("expected inspector status " +
            std::to_string(expected) + ", got " + std::to_string(error.status()) + ": " + error.what());
        return;
    }
    throw std::runtime_error("inspector accepted a damaged bundle");
}
}
int main(int argc, char** argv)
{
    try {
        require(argc == 5, "usage: bundle_contracts BUNDLE PRODUCT DEVELOPMENT SCRATCH");
        auto inspected = tensoracle::inspect_bundle(argv[1]);
        auto digest = inspected.sha256();
        require(tensoracle::inspect_bundle(argv[1], argv[2]).sha256() == digest, "product bundle differs");
        require(tensoracle::inspect_bundle(argv[1], argv[3]).sha256() == digest, "development bundle differs");
        fs::path source = argv[1], scratch = argv[4];
        // CTest gives this test a private disposable directory under its build tree.
        fs::remove_all(scratch);
        fs::copy(source, scratch, fs::copy_options::recursive);
        struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{scratch};
        auto cubin = scratch / "cubins/sm120/fc1_fp16.cubin";
        {
            std::fstream file(cubin, std::ios::binary | std::ios::in | std::ios::out);
            char value{};
            file.seekg(256); file.read(&value, 1); value ^= 0x01;
            file.seekp(256); file.write(&value, 1);
        }
        rejected(scratch, TENSORACLE_CORRUPT_CUBIN);
        fs::remove(cubin);
        rejected(scratch, TENSORACLE_MISSING_CUBIN);
        fs::copy_file(source / "cubins/sm120/fc1_fp16.cubin", cubin);
        auto manifest_path = scratch / "bundle.json";
        std::ifstream input(manifest_path);
        std::string manifest{std::istreambuf_iterator<char>(input), {}};
        auto position = manifest.find("\"target_sm\": \"sm120\"");
        require(position != std::string::npos, "manifest lost sm120");
        manifest.replace(position, std::string("\"target_sm\": \"sm120\"").size(), "\"target_sm\": \"sm89\"");
        std::ofstream(manifest_path) << manifest;
        // Recompute the manifest digest so a false SM reaches architecture validation.
        auto replacement = tensoracle::bundle_digest(manifest.c_str()).sha256();
        position = manifest.find(digest);
        require(position != std::string::npos, "manifest lost bundle digest");
        manifest.replace(position, digest.size(), replacement);
        std::ofstream(manifest_path) << manifest;
        rejected(scratch, TENSORACLE_CUBIN_ARCHITECTURE_ERROR);
        std::cout << "bundle_sha256=" << digest << ": embedded bytes, corruption, missing image and SM checks pass\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
