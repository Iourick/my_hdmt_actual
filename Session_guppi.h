#pragma once
#include "SessionB.h"
class CSession_guppi :
    public CSessionB
{
public:
	CSession_guppi();
	CSession_guppi(const  CSession_guppi& R);
	CSession_guppi& operator=(const CSession_guppi& R);
	CSession_guppi(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft);
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
	virtual bool openFileReadingStream(FILE**& prb_File);
	
	virtual int calcQuantBlocks(unsigned long long* pilength);

	virtual bool readTelescopeHeader(FILE* r_file
		, int* nbits
		, float* chanBW
		, int* npol
		, bool* bdirectIO
		, float* centfreq
		, int* nchan
		, float* obsBW
		, long long* nblocksize
		, EN_telescope* TELESCOP
		, float* tresolution
	);

	virtual bool createCurrentTelescopeHeader(FILE** prb_File);

	virtual size_t download_chunk(FILE** rb_File, char* d_parrInput, const long long QUantDownloadingBytes);

	virtual bool  unpack_chunk(const long long lenChunk, const int j
		, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);

	virtual bool closeFileReadingStream(FILE**& prb_File);

	virtual bool allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
		, const int QUantChunkComplexNumbers);

	virtual void createChunk(CChunkB** ppchunk
		, const float Fmin
		, const float Fmax
		, const int npol
		, const int nchan		
		, const unsigned int len_sft
		, const int Block_id
		, const int Chunk_id
		, const double d_max
		, const double d_min
		, const int ncoherent
		, const float sigma_bound
		, const int length_sum_wnd
		, const int nbin
		, const int nfft
		, const int noverlap
		, const float tsamp);

	virtual void freeInputMemory(void* parrInput, void* pcmparrRawSignalCur);

	virtual void  shift_file_pos(FILE** prb_File, const int IShift);

	virtual bool navigateToBlock(FILE* rb_File, const int IBlockNum);
	
	virtual size_t  calc_ShiftingBytes(unsigned int QUantChunkBytes);
	
};

