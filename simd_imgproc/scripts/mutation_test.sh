#!/bin/bash
# Mutation testing: deliberately break the code in one small way, rebuild, and check that the test-suite notices.
# A "survivor" (mutant that passes all tests) would mean a hole in the tests.  Usage: scripts/mutation_test.sh > results/mutation_results.txt
# Each line:  <status> | <file> | <what was changed> | <how detected>
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

mutate() {   # file  sed-expression  description  [asan]
    local file="$1" expr="$2" desc="$3" mode="${4:-plain}"
    rm -rf "$W/w"; mkdir -p "$W/w"
    cp -r "$ROOT"/{include,src,apps,tests,Makefile} "$W/w/"
    (cd "$W/w" && sed -i "$expr" "$file")
    if cmp -s "$W/w/$file" "$ROOT/$file"; then echo "ERROR | $file | $desc | mutation did not apply"; return; fi
    local out
    if [[ "$mode" == "asan" ]]; then
        out=$(cd "$W/w" && make asan-test 2>&1)
        if echo "$out" | grep -q "ERROR: AddressSanitizer"; then echo "KILLED | $file | $desc | AddressSanitizer"; return; fi
    else
        (cd "$W/w" && make -j4 build/simdip_tests >/dev/null 2>&1) || { echo "ERROR | $file | $desc | build failed"; return; }
        out=$(cd "$W/w" && ./build/simdip_tests 2>&1 | tail -1)
    fi
    if echo "$out" | grep -q " 0 failures"; then echo "SURVIVED | $file | $desc | -"; else echo "KILLED | $file | $desc | $(echo "$out" | sed 's/.* checks, //')"; fi
}

mutate src/avx2.cpp 's/_mm256_add_epi16(dxb, dxb)/dxb/' "Sobel: drop the weight 2 of the middle row"
mutate src/avx2.cpp 's/_mm256_set1_epi16(150)/_mm256_set1_epi16(151)/' "gray: green coefficient 150 -> 151"
mutate src/avx2.cpp 's/x + 16 <= w - 1; x += 16/x + 16 <= w; x += 16/' "Sobel: vector loop reads one byte past the row (result silently repaired by the scalar tail)" asan
mutate src/avx2.cpp '0,/w - x));/s//w - x + 1));/' "conv: copy one byte too many for the last partial block (writes into row padding)"
mutate src/avx2.cpp 's/if (!needed) slot = s;/if (true) slot = s;/' "conv: ring cache may evict a row that is still needed"
mutate src/avx2.cpp 's/_mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)/_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)/' "conv: omit the lane-order fix after packus"
mutate src/avx2.cpp 's/_mm256_cvtps_epi32(acc\[4 \* g\])/_mm256_cvttps_epi32(acc[4 * g])/' "conv: truncate instead of round (first vector of each group)"
mutate src/scalar_impl.inc 's/q\[w - 1\] = sobelPixel(a, b, c, w - 2, w - 1, w - 1);/q[w - 1] = sobelPixel(a, b, c, w - 2, w - 1, w - 2);/' "scalar Sobel: wrong right-border neighbour"
mutate src/scalar_impl.inc 's/(r > 255 ? 255 : r)/(r > 254 ? 254 : r)/' "scalar conv: clamp at 254 instead of 255"
mutate src/scalar_impl.inc 's/ + 128u) >> 8/ + 127u) >> 8/' "scalar gray: rounding constant 128 -> 127"
mutate src/dispatch.cpp 's/int y1 = static_cast<int>(static_cast<long long>(h) \* (t + 1) \/ n);/int y1 = static_cast<int>(static_cast<long long>(h) * (t + 1) \/ n) - (t + 1 == n ? 1 : 0);/' "OpenMP: last band loses its final row"
mutate src/image.cpp 's/if (s % 4096 == 0) s += 64;/if (false) s += 64;/' "Image: remove the page-multiple stride guard"
