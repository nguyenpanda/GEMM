#ifndef DEBUG_NEW_H
#define DEBUG_NEW_H

#ifdef NP_ENABLE_MEMEAK_DEBUG

#include <iostream>
#include <map>
#include <cstdlib>

#include "nguyenpanda.h"

struct LeakInfo {
    const char* file;
    int line;
    size_t size;
};

static std::map<void*, LeakInfo> leak_map;

inline void* operator new(size_t size, const char* file, int line) {
    void* p = std::malloc(size);
    if (!p) {
        std::cerr << "Memory allocation failed at " << file << ":" << line << std::endl;
        std::abort();
    }
	printf("Allocated %7zu bytes at %p (%s:%d)\n", size, p, file, line);
    leak_map[p] = {file, line, size};
    return p;
}

inline void* operator new[](size_t size, const char* file, int line) {
    void* p = std::malloc(size);
    if (!p) {
        std::cerr << "Memory allocation failed at " << file << ":" << line << std::endl;
        std::abort();
    }
    leak_map[p] = {file, line, size};
	printf("Allocated %7zu bytes at %p (%s:%d)\n", size, p, file, line);
    return p;
}

void operator delete(void* p) noexcept {
    auto it = leak_map.find(p);
	
    if (it != leak_map.end()) {
		printf(CYAN "Deleting address at %p\n" RESET, p);
        leak_map.erase(it);
        std::free(p);
    } else {
		printf(YELLOW "MisDelete address at %p\n" RESET, p);
    }
}

void operator delete[](void* p) noexcept {
    auto it = leak_map.find(p);
    if (it != leak_map.end()) {
		printf(CYAN "Deleting address %p\n" RESET, p);
        leak_map.erase(it);
        std::free(p);
    } else {
        printf(YELLOW "MisDelete address at %p\n" RESET, p);
    }
}

void check_leaks(FILE* file) {
    if (!leak_map.empty()) {
		fprintf(file, "object_size,address,file,line\n");
        for (const auto& entry : leak_map) {
			auto sec = entry.second;
			fprintf(file, "%zu,%p,%s,%d\n", 
				sec.size, entry.first, sec.file, sec.line
			);
        }
    }
}

void check_leaks() {
	check_leaks(stdout);
}

void check_leaks(const char* file_name) {
	FILE* file_ptr = fopen(file_name, "w");
	if (!file_ptr) {
		printf("Can't open file `%s`\n", file_name);
		check_leaks(stdout);
	}
	check_leaks(file_ptr);
	fclose(file_ptr);
}

#define DEBUG_NEW new(__FILE__, __LINE__)
#define new DEBUG_NEW

#endif // NP_ENABLE_MEMEAK_DEBUG

#endif // DEBUG_NEW_H
