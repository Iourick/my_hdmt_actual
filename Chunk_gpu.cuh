#pragma once
#include "stdio.h"
#include <cufft.h>
#include "Constants.h"
#include <complex> 
#include "FdmtGpu.cuh"
#include "ChunkB.h"
#define TILE_DIM 32
using namespace std;

class COutChunkHeader;

class CFdmtU;
class CTelescopeHeader;
 
class CChunk_gpu : public CChunkB
{
public:
	~CChunk_gpu();
	CChunk_gpu();
	CChunk_gpu(const  CChunk_gpu& R);
	CChunk_gpu& operator=(const CChunk_gpu& R);
	CChunk_gpu(
		const float Fmin
		, const float Fmax
		, const int npol
		, const int nchan
		, const unsigned int len_sft
		, const int Block_id
		, const int Chunk_id
		, const  float d_max
		, const  float d_min
		, const int ncoherent
		, const float sigma_bound
		, const int length_sum_wnd
		, const int nbin
		, const int nfft
		, const int noverlap
		, const float tsamp);
	//---------------------------------------------------------------------------------------
	double* m_pd_arrcoh_dm;	// =m_coh_dm_Vector.data()

	cufftHandle m_fftPlanForward;  // plan for cufft

	cufftHandle m_fftPlanInverse; // plan for cufft

	CFdmtGpu m_Fdmt; //FDMT

	//cufftComplex  buffer to store element wize multed array
	cufftComplex* m_pdcmpbuff_ewmulted;

	// buffer to store rolled array
	float* m_pdbuff_rolled;

	// buffer to accumulate fdmt outputs
	float* m_pdoutImg;	

	//-------------------------------------------------------------------------
	virtual bool process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg);	

	void set_chunkid(const int nC);

	void set_blockid(const int nC);	

	virtual void compute_chirp_channel();

	void create_fft_plans();

	virtual void  elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, int  idm);

};
__global__
void kernel_create_arr_freqs_chan(double* d_parr_freqs_chan, int len_sft, double bw_chan, double  Fmin, double bw_sub);

__global__
void kernel_create_arr_bin_freqs_and_taper(double* d_parr_bin_freqs, double* d_parr_taper, double  bw_chan, int mbin);

__global__ 
void roll_rows_and_normalize_kernel(cufftComplex* arr_rez, cufftComplex* arr, int rows, int cols, int shift);

__global__ 
void  element_wise_cufftComplex_mult_kernel(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, cufftComplex* d_arrInp1
	, int npol, int nfft, int dim2);

__global__ void  divide_cufftComplex_array_kernel(cufftComplex* d_arr, int len, float val);

__global__
void calc_intensity_kernel(float *intensity, const int len, const int npol, cufftComplex* fbuf);

__global__
void  transpose_unpadd_kernel(cufftComplex* fbuf, cufftComplex* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nchan, const int nlen_sft, int mbin);

__global__	void  transpose_unpadd_kernel_(float* fbuf, cufftComplex* arin, int nfft, int npol, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, const int mbin);

__global__ void transpose(float* odata, float* idata, int width, int height);

__global__
void dedisperse(float* parr_wfall_disp, float* parrIntesity, double dm, double fmin, double fmax, double   val_tsamp_wfall, double foff, int cols);

__global__
void scaling_kernel(cufftComplex* data, long long element_count, float scale);

__device__
float fnc_norm2(cufftComplex* pcarr);

__global__
void calcPartSum_kernel(float* d_parr_out, const int lenChunk, const int npol_physical, cufftComplex* d_parr_inp);

__global__
void calcMultiTransposition_kernel(fdmt_type_* output, const int height, const int width, fdmt_type_* input);

__global__
void calcPowerMtrx_kernel(float* output, const int height, const int width, const int npol, cufftComplex* input);

__global__
void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int lenChunk, float* d_pOutArray);

__global__
void multiTransp_kernel(float* output, const int height, const int width, float* input);

__global__ 	void  transpose_unpadd_intensity(float* fbuf, float* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, int mbin);

__global__ void roll_rows_normalize_sum_kernel(float* arr_rez, cufftComplex* arr, const int npol, const int rows
	, const  int cols, const  int shift);




