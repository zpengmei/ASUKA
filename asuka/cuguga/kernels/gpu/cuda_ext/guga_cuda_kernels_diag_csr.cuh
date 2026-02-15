__global__ void build_occ_block_from_steps_kernel(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    double* __restrict__ occ_out) {  // [j_count,norb]
  int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  int total = j_count * norb;
  if (t >= total) return;

  int j_local = t / norb;
  int k = t - j_local * norb;
  int csf_idx = j_start + j_local;
  if ((unsigned)csf_idx >= (unsigned)ncsf) return;

  int8_t step = steps_table[(int64_t)csf_idx * (int64_t)norb + (int64_t)k];
  occ_out[(int64_t)j_local * (int64_t)norb + (int64_t)k] = (double)step_to_occ(step);
}

extern "C" cudaError_t guga_build_occ_block_from_steps_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    double* occ_out,
    cudaStream_t stream,
    int threads) {
  if (!steps_table || !occ_out) return cudaErrorInvalidValue;
  if (ncsf < 0 || norb < 0 || j_start < 0 || j_count < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (j_count == 0 || norb == 0) return cudaSuccess;
  if (j_start + j_count > ncsf) return cudaErrorInvalidValue;

  int64_t total = (int64_t)j_count * (int64_t)norb;
  int blocks = (int)((total + (int64_t)threads - 1) / (int64_t)threads);
  build_occ_block_from_steps_kernel<<<blocks, threads, 0, stream>>>(steps_table, ncsf, norb, j_start, j_count, occ_out);
  return cudaGetLastError();
}

template <typename T>
__global__ void build_w_diag_from_steps_kernel_t(
    const int8_t* __restrict__ steps_table,  // [ncsf,norb]
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const T* __restrict__ x,  // [ncsf]
    int nops,
    T* __restrict__ w_out,  // [ncsf,w_stride]
    int64_t w_stride,
    int relative_w) {
  int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  int total = j_count * norb;
  if (t >= total) return;

  int j_local = t / norb;
  int r = t - j_local * norb;
  int csf_idx = j_start + j_local;
  if ((unsigned)csf_idx >= (unsigned)ncsf) return;

  T scale = x[csf_idx];
  if (scale == (T)0) return;

  int8_t step = steps_table[(int64_t)csf_idx * (int64_t)norb + (int64_t)r];
  int occ_r = step_to_occ(step);
  if (!occ_r) return;

  int rr = r * norb + r;
  if ((unsigned)rr >= (unsigned)nops) return;
  
  int64_t row_idx = (int64_t)(relative_w ? j_local : csf_idx);
  w_out[row_idx * w_stride + (int64_t)rr] = scale * (T)occ_r;
}

extern "C" cudaError_t guga_build_w_diag_from_steps_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const double* x,
    int nops,
    double* w_out,
    int64_t w_stride,
    cudaStream_t stream,
  int threads,
  int relative_w) {
  if (!steps_table || !x || !w_out) return cudaErrorInvalidValue;
  if (ncsf < 0 || norb < 0 || j_start < 0 || j_count < 0) return cudaErrorInvalidValue;
  if (nops != norb * norb) return cudaErrorInvalidValue;
  if (w_stride < (int64_t)nops) return cudaErrorInvalidValue;
  if (j_start + j_count > ncsf) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (j_count <= 0) return cudaSuccess;

  int total = j_count * norb;
  int blocks = (total + threads - 1) / threads;
  build_w_diag_from_steps_kernel_t<double><<<blocks, threads, 0, stream>>>(
      steps_table, ncsf, norb, j_start, j_count, x, nops, w_out, w_stride, relative_w);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_build_w_diag_from_steps_f32_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    int j_start,
    int j_count,
    const float* x,
    int nops,
    float* w_out,
    int64_t w_stride,
    cudaStream_t stream,
    int threads,
    int relative_w) {
  if (!steps_table || !x || !w_out) return cudaErrorInvalidValue;
  if (ncsf < 0 || norb < 0 || j_start < 0 || j_count < 0) return cudaErrorInvalidValue;
  if (nops != norb * norb) return cudaErrorInvalidValue;
  if (w_stride < (int64_t)nops) return cudaErrorInvalidValue;
  if (j_start + j_count > ncsf) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (j_count <= 0) return cudaSuccess;

  int total = j_count * norb;
  int blocks = (total + threads - 1) / threads;
  build_w_diag_from_steps_kernel_t<float><<<blocks, threads, 0, stream>>>(
      steps_table, ncsf, norb, j_start, j_count, x, nops, w_out, w_stride, relative_w);
  return cudaGetLastError();
}

__global__ void build_hdiag_det_guess_from_steps_kernel(
    const int8_t* __restrict__ steps_table,      // [ncsf,norb]
    int ncsf,                                    //
    int norb,                                    //
    int neleca_det,                              //
    const double* __restrict__ h1e_diag,         // [norb]
    const double* __restrict__ eri_ppqq,         // [norb,norb]
    const double* __restrict__ eri_pqqp,         // [norb,norb]
    double* __restrict__ hdiag_out) {            // [ncsf]
  int csf = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if ((unsigned)csf >= (unsigned)ncsf) return;

  // Build doubly/single masks + counts from the cached steps table.
  uint64_t mask_doubly = 0;
  uint64_t mask_single = 0;
  int ndoubly = 0;
  int nsingle = 0;
  for (int k = 0; k < norb; k++) {
    int8_t step = steps_table[(int64_t)csf * (int64_t)norb + (int64_t)k];
    if (step == 3) {
      mask_doubly |= (uint64_t(1) << (unsigned)k);
      ndoubly++;
    } else if (step == 1 || step == 2) {
      mask_single |= (uint64_t(1) << (unsigned)k);
      nsingle++;
    }
  }

  int alpha_need = neleca_det - ndoubly;
  if (alpha_need < 0) alpha_need = 0;
  if (alpha_need > nsingle) alpha_need = nsingle;

  uint64_t mask_alpha_single = 0;
  uint64_t mask_beta_single = 0;
  int seen = 0;
  for (int k = 0; k < norb; k++) {
    uint64_t bit = (uint64_t(1) << (unsigned)k);
    if (!(mask_single & bit)) continue;
    if (seen < alpha_need) mask_alpha_single |= bit;
    else mask_beta_single |= bit;
    seen++;
  }
  uint64_t mask_alpha = mask_doubly | mask_alpha_single;
  uint64_t mask_beta = mask_doubly | mask_beta_single;

  // One-body term.
  double acc = 0.0;
  for (int p = 0; p < norb; p++) {
    uint64_t bit = (uint64_t(1) << (unsigned)p);
    int n_p = (mask_doubly & bit) ? 2 : ((mask_single & bit) ? 1 : 0);
    if (n_p) acc += (double)n_p * h1e_diag[p];
  }

  // Coulomb term: 0.5 * sum_{p,q} n_p * n_q * (pp|qq).
  double coul = 0.0;
  for (int p = 0; p < norb; p++) {
    uint64_t bit_p = (uint64_t(1) << (unsigned)p);
    int n_p = (mask_doubly & bit_p) ? 2 : ((mask_single & bit_p) ? 1 : 0);
    if (!n_p) continue;
    for (int q = 0; q < norb; q++) {
      uint64_t bit_q = (uint64_t(1) << (unsigned)q);
      int n_q = (mask_doubly & bit_q) ? 2 : ((mask_single & bit_q) ? 1 : 0);
      if (!n_q) continue;
      coul += 0.5 * (double)n_p * (double)n_q * eri_ppqq[(int64_t)p * (int64_t)norb + (int64_t)q];
    }
  }
  acc += coul;

  // Exchange terms: -0.5 * sum_{p,q} alpha_p*alpha_q*(pq|qp) and same for beta.
  double exa = 0.0;
  for (int p = 0; p < norb; p++) {
    uint64_t bit_p = (uint64_t(1) << (unsigned)p);
    if (!(mask_alpha & bit_p)) continue;
    for (int q = 0; q < norb; q++) {
      uint64_t bit_q = (uint64_t(1) << (unsigned)q);
      if (!(mask_alpha & bit_q)) continue;
      exa += -0.5 * eri_pqqp[(int64_t)p * (int64_t)norb + (int64_t)q];
    }
  }
  double exb = 0.0;
  for (int p = 0; p < norb; p++) {
    uint64_t bit_p = (uint64_t(1) << (unsigned)p);
    if (!(mask_beta & bit_p)) continue;
    for (int q = 0; q < norb; q++) {
      uint64_t bit_q = (uint64_t(1) << (unsigned)q);
      if (!(mask_beta & bit_q)) continue;
      exb += -0.5 * eri_pqqp[(int64_t)p * (int64_t)norb + (int64_t)q];
    }
  }
  acc += exa + exb;
  hdiag_out[csf] = acc;
}

extern "C" cudaError_t guga_build_hdiag_det_guess_from_steps_launch_stream(
    const int8_t* steps_table,
    int ncsf,
    int norb,
    int neleca_det,
    const double* h1e_diag,
    const double* eri_ppqq,
    const double* eri_pqqp,
    double* hdiag_out,
    cudaStream_t stream,
    int threads) {
  if (!steps_table || !h1e_diag || !eri_ppqq || !eri_pqqp || !hdiag_out) return cudaErrorInvalidValue;
  if (ncsf < 0 || norb < 0) return cudaErrorInvalidValue;
  if (norb > 64) return cudaErrorInvalidValue;
  if (neleca_det < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (ncsf == 0 || norb == 0) return cudaSuccess;

  int blocks = (int)(((int64_t)ncsf + (int64_t)threads - 1) / (int64_t)threads);
  build_hdiag_det_guess_from_steps_kernel<<<blocks, threads, 0, stream>>>(
      steps_table, ncsf, norb, neleca_det, h1e_diag, eri_ppqq, eri_pqqp, hdiag_out);
  return cudaGetLastError();
}

template <typename T>
__global__ void build_g_from_csr_eri_mat_kernel_t(
    const int64_t* __restrict__ indptr,   // [nrows+1]
    const int32_t* __restrict__ indices,  // [nnz]
    const T* __restrict__ data,           // [nnz]
    const T* __restrict__ eri_mat,        // [nops,nops] row-major, eri_mat[pq*nops + rs]
    int nops,
    T half,
    T* __restrict__ g_out) {  // [nrows,nops] row-major
  int row = (int)blockIdx.x;
  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];

  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    T acc = (T)0;
    for (int64_t t = start; t < end; t++) {
      int32_t rs = indices[t];
      T c = data[t];
      acc += eri_mat[(int64_t)pq * (int64_t)nops + (int64_t)rs] * c;
    }
    g_out[(int64_t)row * (int64_t)nops + (int64_t)pq] = half * acc;
  }
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int nrows,
    const double* eri_mat,
    int nops,
    double half,
    double* g_out,
    cudaStream_t stream,
    int threads);

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_f64_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int nrows,
    const double* eri_mat,
    int nops,
    double half,
    double* g_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !eri_mat || !g_out) return cudaErrorInvalidValue;
  if (nrows < 0 || nops < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || nops == 0) return cudaSuccess;

  build_g_from_csr_eri_mat_kernel_t<double><<<nrows, threads, 0, stream>>>(
      indptr, indices, data, eri_mat, nops, half, g_out);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_f32_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const float* data,
    int nrows,
    const float* eri_mat,
    int nops,
    float half,
    float* g_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !eri_mat || !g_out) return cudaErrorInvalidValue;
  if (nrows < 0 || nops < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || nops == 0) return cudaSuccess;

  build_g_from_csr_eri_mat_kernel_t<float><<<nrows, threads, 0, stream>>>(
      indptr, indices, data, eri_mat, nops, half, g_out);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_launch(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int nrows,
    const double* eri_mat,
    int nops,
    double half,
    double* g_out,
    int threads) {
  return guga_build_g_from_csr_eri_mat_f64_launch_stream(
      indptr, indices, data, nrows, eri_mat, nops, half, g_out, /*stream=*/0, threads);
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int nrows,
    const double* eri_mat,
    int nops,
    double half,
    double* g_out,
    cudaStream_t stream,
    int threads) {
  return guga_build_g_from_csr_eri_mat_f64_launch_stream(
      indptr, indices, data, nrows, eri_mat, nops, half, g_out, stream, threads);
}

template <typename T>
__global__ void build_g_from_csr_eri_mat_range_kernel_t(
    const int64_t* __restrict__ indptr,   // [nrows_total+1]
    const int32_t* __restrict__ indices,  // [nnz_total]
    const T* __restrict__ data,           // [nnz_total]
    int row_start,
    int nrows,
    const T* __restrict__ eri_mat,       // [nops,nops] row-major, eri_mat[pq*nops + rs]
    int nops,
    T half,
    T* __restrict__ g_out) {  // [nrows,nops] row-major
  int row_local = (int)blockIdx.x;
  if (row_local >= nrows) return;
  int row = row_start + row_local;

  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];

  for (int pq = (int)threadIdx.x; pq < nops; pq += (int)blockDim.x) {
    T acc = (T)0;
    for (int64_t t = start; t < end; t++) {
      int32_t rs = indices[t];
      T c = data[t];
      acc += eri_mat[(int64_t)pq * (int64_t)nops + (int64_t)rs] * c;
    }
    g_out[(int64_t)row_local * (int64_t)nops + (int64_t)pq] = half * acc;
  }
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_range_f64_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int row_start,
    int nrows,
    const double* eri_mat,
    int nops,
    double half,
    double* g_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !eri_mat || !g_out) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || nops < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || nops == 0) return cudaSuccess;

  build_g_from_csr_eri_mat_range_kernel_t<double><<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, eri_mat, nops, half, g_out);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_range_f32_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const float* data,
    int row_start,
    int nrows,
    const float* eri_mat,
    int nops,
    float half,
    float* g_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !eri_mat || !g_out) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || nops < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || nops == 0) return cudaSuccess;

  build_g_from_csr_eri_mat_range_kernel_t<float><<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, eri_mat, nops, half, g_out);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_build_g_from_csr_eri_mat_range_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int row_start,
    int nrows,
    const double* eri_mat,
    int nops,
    double half,
    double* g_out,
    cudaStream_t stream,
    int threads) {
  return guga_build_g_from_csr_eri_mat_range_f64_launch_stream(
      indptr, indices, data, row_start, nrows, eri_mat, nops, half, g_out, stream, threads);
}

__global__ void csr_to_dense_f64_kernel(
    const int64_t* __restrict__ indptr,   // [nrows+1]
    const int32_t* __restrict__ indices,  // [nnz]
    const double* __restrict__ data,      // [nnz]
    int nrows,
    int ncols,
    double* __restrict__ out_dense) {  // [nrows,ncols] row-major
  int row = (int)blockIdx.x;
  if (row >= nrows) return;
  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];
  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t col = indices[t];
    double v = data[t];
    if (col >= 0 && col < ncols) {
      // Use atomicAdd to be robust to duplicate indices (should not happen after coalesce).
      atomicAdd(&out_dense[(int64_t)row * (int64_t)ncols + (int64_t)col], v);
    }
  }
}

__global__ void csr_to_dense_f64_range_kernel(
    const int64_t* __restrict__ indptr,   // [nrows_total+1]
    const int32_t* __restrict__ indices,  // [nnz_total]
    const double* __restrict__ data,      // [nnz_total]
    int row_start,
    int nrows,
    int ncols,
    double* __restrict__ out_dense) {  // [nrows,ncols] row-major
  int row = (int)blockIdx.x;
  if (row >= nrows) return;
  int64_t start = indptr[(int64_t)row_start + (int64_t)row];
  int64_t end = indptr[(int64_t)row_start + (int64_t)row + 1];
  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t col = indices[t];
    double v = data[t];
    if (col >= 0 && col < ncols) {
      atomicAdd(&out_dense[(int64_t)row * (int64_t)ncols + (int64_t)col], v);
    }
  }
}

extern "C" cudaError_t guga_csr_to_dense_f64_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int nrows,
    int ncols,
    double* out_dense,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !out_dense) return cudaErrorInvalidValue;
  if (nrows < 0 || ncols < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || ncols == 0) return cudaSuccess;
  csr_to_dense_f64_kernel<<<nrows, threads, 0, stream>>>(indptr, indices, data, nrows, ncols, out_dense);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_csr_to_dense_f64_range_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int row_start,
    int nrows,
    int ncols,
    double* out_dense,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !out_dense) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || ncols < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || ncols == 0) return cudaSuccess;
  csr_to_dense_f64_range_kernel<<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, ncols, out_dense);
  return cudaGetLastError();
}

__global__ void csr_to_dense_f32_kernel(
    const int64_t* __restrict__ indptr,   // [nrows+1]
    const int32_t* __restrict__ indices,  // [nnz]
    const float* __restrict__ data,       // [nnz]
    int nrows,
    int ncols,
    float* __restrict__ out_dense) {  // [nrows,ncols] row-major
  int row = (int)blockIdx.x;
  if (row >= nrows) return;
  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];
  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t col = indices[t];
    float v = data[t];
    if (col >= 0 && col < ncols) {
      // Use atomicAdd to be robust to duplicate indices (should not happen after coalesce).
      atomicAdd(&out_dense[(int64_t)row * (int64_t)ncols + (int64_t)col], v);
    }
  }
}

__global__ void csr_to_dense_f32_range_kernel(
    const int64_t* __restrict__ indptr,   // [nrows_total+1]
    const int32_t* __restrict__ indices,  // [nnz_total]
    const float* __restrict__ data,       // [nnz_total]
    int row_start,
    int nrows,
    int ncols,
    float* __restrict__ out_dense) {  // [nrows,ncols] row-major
  int row = (int)blockIdx.x;
  if (row >= nrows) return;
  int64_t start = indptr[(int64_t)row_start + (int64_t)row];
  int64_t end = indptr[(int64_t)row_start + (int64_t)row + 1];
  for (int64_t t = start + (int64_t)threadIdx.x; t < end; t += (int64_t)blockDim.x) {
    int32_t col = indices[t];
    float v = data[t];
    if (col >= 0 && col < ncols) {
      atomicAdd(&out_dense[(int64_t)row * (int64_t)ncols + (int64_t)col], v);
    }
  }
}

extern "C" cudaError_t guga_csr_to_dense_f32_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const float* data,
    int nrows,
    int ncols,
    float* out_dense,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !out_dense) return cudaErrorInvalidValue;
  if (nrows < 0 || ncols < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || ncols == 0) return cudaSuccess;
  csr_to_dense_f32_kernel<<<nrows, threads, 0, stream>>>(indptr, indices, data, nrows, ncols, out_dense);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_csr_to_dense_f32_range_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const float* data,
    int row_start,
    int nrows,
    int ncols,
    float* out_dense,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !out_dense) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || ncols < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || ncols == 0) return cudaSuccess;
  csr_to_dense_f32_range_kernel<<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, ncols, out_dense);
  return cudaGetLastError();
}

__global__ void csr_l_full_to_wt_f64_range_kernel(
    const int64_t* __restrict__ indptr,   // [nrows_total+1]
    const int32_t* __restrict__ indices,  // [nnz_total]
    const double* __restrict__ data,      // [nnz_total]
    int row_start,
    int nrows,
    const double* __restrict__ l_full,  // [nops,naux] row-major
    int naux,
    double* __restrict__ wt_out) {  // [naux,nrows] column-major (ld=naux)
  int row_local = (int)blockIdx.x;
  if (row_local >= nrows) return;
  int row = row_start + row_local;
  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];

  for (int l = (int)threadIdx.x; l < naux; l += (int)blockDim.x) {
    double acc = 0.0;
    for (int64_t t = start; t < end; t++) {
      int32_t rs = indices[t];
      double c = data[t];
      acc += c * l_full[(int64_t)rs * (int64_t)naux + (int64_t)l];
    }
    wt_out[(int64_t)l + (int64_t)row_local * (int64_t)naux] = acc;
  }
}

extern "C" cudaError_t guga_csr_l_full_to_wt_f64_range_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int row_start,
    int nrows,
    const double* l_full,
    int naux,
    double* wt_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !l_full || !wt_out) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || naux < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || naux == 0) return cudaSuccess;
  csr_l_full_to_wt_f64_range_kernel<<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, l_full, naux, wt_out);
  return cudaGetLastError();
}

__global__ void csr_l_full_to_wt_f32_range_kernel(
    const int64_t* __restrict__ indptr,   // [nrows_total+1]
    const int32_t* __restrict__ indices,  // [nnz_total]
    const float* __restrict__ data,       // [nnz_total]
    int row_start,
    int nrows,
    const float* __restrict__ l_full,  // [nops,naux] row-major
    int naux,
    float* __restrict__ wt_out) {  // [naux,nrows] column-major (ld=naux)
  int row_local = (int)blockIdx.x;
  if (row_local >= nrows) return;
  int row = row_start + row_local;
  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];

  for (int l = (int)threadIdx.x; l < naux; l += (int)blockDim.x) {
    float acc = 0.0f;
    for (int64_t t = start; t < end; t++) {
      int32_t rs = indices[t];
      float c = data[t];
      acc = fmaf(c, l_full[(int64_t)rs * (int64_t)naux + (int64_t)l], acc);
    }
    wt_out[(int64_t)l + (int64_t)row_local * (int64_t)naux] = acc;
  }
}

extern "C" cudaError_t guga_csr_l_full_to_wt_f32_range_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const float* data,
    int row_start,
    int nrows,
    const float* l_full,
    int naux,
    float* wt_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !l_full || !wt_out) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || naux < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || naux == 0) return cudaSuccess;
  csr_l_full_to_wt_f32_range_kernel<<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, l_full, naux, wt_out);
  return cudaGetLastError();
}

__global__ void csr_l_full_to_wt_f32_from_f64_range_kernel(
    const int64_t* __restrict__ indptr,   // [nrows_total+1]
    const int32_t* __restrict__ indices,  // [nnz_total]
    const double* __restrict__ data,      // [nnz_total]
    int row_start,
    int nrows,
    const float* __restrict__ l_full,  // [nops,naux] row-major
    int naux,
    float* __restrict__ wt_out) {  // [naux,nrows] column-major (ld=naux)
  int row_local = (int)blockIdx.x;
  if (row_local >= nrows) return;
  int row = row_start + row_local;
  int64_t start = indptr[row];
  int64_t end = indptr[row + 1];

  for (int l = (int)threadIdx.x; l < naux; l += (int)blockDim.x) {
    float acc = 0.0f;
    for (int64_t t = start; t < end; t++) {
      int32_t rs = indices[t];
      float c = (float)data[t];
      acc = fmaf(c, l_full[(int64_t)rs * (int64_t)naux + (int64_t)l], acc);
    }
    wt_out[(int64_t)l + (int64_t)row_local * (int64_t)naux] = acc;
  }
}

extern "C" cudaError_t guga_csr_l_full_to_wt_f32_from_f64_range_launch_stream(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int row_start,
    int nrows,
    const float* l_full,
    int naux,
    float* wt_out,
    cudaStream_t stream,
    int threads) {
  if (!indptr || !indices || !data || !l_full || !wt_out) return cudaErrorInvalidValue;
  if (row_start < 0 || nrows < 0 || naux < 0) return cudaErrorInvalidValue;
  if (threads <= 0 || threads > 1024) return cudaErrorInvalidValue;
  if (nrows == 0 || naux == 0) return cudaSuccess;
  csr_l_full_to_wt_f32_from_f64_range_kernel<<<nrows, threads, 0, stream>>>(
      indptr, indices, data, row_start, nrows, l_full, naux, wt_out);
  return cudaGetLastError();
}

extern "C" cudaError_t guga_csr_to_dense_f64_launch(
    const int64_t* indptr,
    const int32_t* indices,
    const double* data,
    int nrows,
    int ncols,
    double* out_dense,
    int threads) {
  return guga_csr_to_dense_f64_launch_stream(indptr, indices, data, nrows, ncols, out_dense, /*stream=*/0, threads);
}
