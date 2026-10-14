// Calls f(y0, y1) for contiguous bands of rows. With threads <= 1 (or no OpenMP) it is one plain call.
// Band t of n covers rows [h*t/n, h*(t+1)/n): sizes differ by at most one row, each thread works on
// memory that is contiguous, and the same thread count always gives the same split.
template <class F>
void forBands(int h, int threads, F f) {
#ifdef _OPENMP
    if (threads > 1 && h > 1) {
#pragma omp parallel num_threads(threads)
        {
            const int t = omp_get_thread_num(), n = omp_get_num_threads();
            const int y0 = static_cast<int>(static_cast<long long>(h) * t / n);
            const int y1 = static_cast<int>(static_cast<long long>(h) * (t + 1) / n);
            if (y0 < y1) f(y0, y1);
        }
        return;
    }
#else
    (void)threads;
#endif
    f(0, h);
}
