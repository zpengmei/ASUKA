#ifndef CUGUGA_CPU_OPENMP_COMPAT_H
#define CUGUGA_CPU_OPENMP_COMPAT_H

#ifdef GUGA_USE_OPENMP
#include <omp.h>
#endif

static int guga_have_openmp(void) {
#ifdef GUGA_USE_OPENMP
  return 1;
#else
  return 0;
#endif
}

static int guga_openmp_max_threads(void) {
#ifdef GUGA_USE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static void guga_openmp_set_num_threads(int n) {
#ifdef GUGA_USE_OPENMP
  if (n < 1) n = 1;
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}

#endif  // CUGUGA_CPU_OPENMP_COMPAT_H
