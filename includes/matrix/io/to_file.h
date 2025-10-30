#ifndef MATRIX_IO_TO_FILE_H
#define MATRIX_IO_TO_FILE_H

#include "../viewer.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>

template <class T>
class SaveFile {
    static std::string dtype_name() {
        if constexpr (std::is_same_v<T, float>) return "float32";
        else if constexpr (std::is_same_v<T, double>) return "float64";
        else if constexpr (std::is_same_v<T, int32_t>) return "int32";
        else if constexpr (std::is_same_v<T, int64_t>) return "int64";
        else if constexpr (std::is_same_v<T, uint8_t>) return "uint8";
        else throw std::runtime_error("Unsupported type for SaveFile");
    }

public:
    static void save(const Buffer<T>& buf, const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for writing");

        const char magic[8] = {'B', 'U', 'F', 'F', 'E', 'R', '1', '\0'};
        out.write(magic, 8);

        std::string dtype = dtype_name();
        uint8_t dtype_len = static_cast<uint8_t>(dtype.size());
        out.write(reinterpret_cast<const char*>(&dtype_len), sizeof(uint8_t));
        out.write(dtype.c_str(), dtype_len);

        // Write dimensions
        out.write(reinterpret_cast<const char*>(&buf.rdim), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&buf.cdim), sizeof(size_t));

        // Write raw data
        out.write(reinterpret_cast<const char*>(buf.data), sizeof(T) * buf.rdim * buf.cdim);
    }

    static Buffer<T> load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for reading");

        char magic[8];
        in.read(magic, 8);
        if (std::string(magic, 7) != "BUFFER")
            throw std::runtime_error("Invalid file format");

        uint8_t dtype_len;
        in.read(reinterpret_cast<char*>(&dtype_len), sizeof(uint8_t));
        std::string dtype(dtype_len, '\0');
        in.read(dtype.data(), dtype_len);

        if (dtype != dtype_name())
            throw std::runtime_error("Type mismatch when loading Buffer");

        size_t rdim, cdim;
        in.read(reinterpret_cast<char*>(&rdim), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&cdim), sizeof(size_t));

        Buffer<T> buf(rdim, cdim);
        in.read(reinterpret_cast<char*>(buf.data), sizeof(T) * rdim * cdim);
        return buf;
    }
};


#endif // MATRIX_IO_TO_FILE_H
