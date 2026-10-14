# SIMD image-processing library - build file
#   make              library, CLI, benchmark, tests -> build/
#   make test         build and run the tests
#   make asan-test    same tests built with AddressSanitizer + UBSan (catches out-of-bounds vector loads/stores)
#   make clean
#
# Useful overrides:
#   make OPT="-O3 -march=native" BUILD=build_native       compiler may now use AVX-512/FMA in the autovec build
#   make OPENMP=0                                         build without OpenMP (threads argument is ignored)
BUILD    ?= build
CXX      ?= g++
OPT      ?= -O3
OPENMP   ?= 1
CXXFLAGS ?= -std=c++17 $(OPT) -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  ?=

ifeq ($(OPENMP),1)
CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp
endif

SRC := $(wildcard src/*.cpp)
OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))
LIB := $(BUILD)/libsimdip.a

all: $(LIB) $(BUILD)/simdip $(BUILD)/simdip_bench $(BUILD)/simdip_tests

$(BUILD):
	mkdir -p $(BUILD)

# Per-file flags. The three implementations differ ONLY in these flags / in the source they include:
#   scalar_novec.cpp    auto-vectorizer OFF, no hidden FMA            -> "scalar"
#   scalar_autovec.cpp  auto-vectorizer as per -O3, no hidden FMA      -> "autovec"
#   avx2.cpp            AVX2 + FMA instructions allowed (called only after a run-time CPU check)
$(BUILD)/scalar_novec.o:   EXTRA := -fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off
$(BUILD)/scalar_autovec.o: EXTRA := -ffp-contract=off
$(BUILD)/avx2.o:           EXTRA := -mavx2 -mfma
$(BUILD)/%.o: EXTRA ?=

$(BUILD)/%.o: src/%.cpp src/internal.hpp $(wildcard include/simdip/*.hpp) src/scalar_impl.inc | $(BUILD)
	$(CXX) $(CXXFLAGS) $(EXTRA) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^

$(BUILD)/simdip: apps/simdip_cli.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(BUILD)/simdip_bench: apps/simdip_bench.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(BUILD)/simdip_tests: tests/test_main.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

test: $(BUILD)/simdip_tests
	./$(BUILD)/simdip_tests

asan-test:
	$(MAKE) BUILD=build_asan OPT="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined" \
	        LDFLAGS="-fsanitize=address,undefined" test

# Which loops did GCC vectorize in the auto-vectorized build? (writes docs/snippets/vec_report.txt)
vec-report:
	$(CXX) $(CXXFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c src/scalar_autovec.cpp -o /dev/null 2> $(BUILD)/vec_report_full.txt || true
	grep -c "optimized: loop vectorized" $(BUILD)/vec_report_full.txt || true

clean:
	rm -rf build build_asan build_native

.PHONY: all test asan-test vec-report clean
