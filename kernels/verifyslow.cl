/* 

	verifyslow.cl - Bryan Little 4/2025, montgomery arithmetic by Yves Gallot
	
	verify power/product/prime tables using slow iterative algorithm

*/

// r0 + 2^64 * r1 = a * b
ulong2 mul_wide(const ulong a, const ulong b){
	ulong2 r;
#ifdef __NV_CL_C_VERSION
	const uint a0 = (uint)(a), a1 = (uint)(a >> 32);
	const uint b0 = (uint)(b), b1 = (uint)(b >> 32);
	uint c0 = a0 * b0, c1 = mul_hi(a0, b0), c2, c3;
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a0), "r" (b1), "r" (c1));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c2) : "r" (a0), "r" (b1));
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b1), "r" (c2));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c3) : "r" (a1), "r" (b1));
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a1), "r" (b0), "r" (c1));
	asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b0), "r" (c2));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (c3) : "r" (c3));
	r.s0 = upsample(c1, c0); r.s1 = upsample(c3, c2);
#else
	r.s0 = a * b; r.s1 = mul_hi(a, b);
#endif
	return r;
}

ulong m_mul(ulong a, ulong b, ulong p, ulong q){
	ulong2 ab = mul_wide(a,b);
	ulong m = ab.s0 * q;
	ulong mp = mul_hi(m,p);
	ulong r = ab.s1 - mp;
	return ( ab.s1 < mp ) ? r + p : r;
}

ulong add(ulong a, ulong b, ulong p){
	ulong r;
	ulong c = (a >= p - b) ? p : 0;
	r = a + b - c;
	return r;
}

// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo
__constant ulong8 prime = (ulong8)(18446744073709551557UL, 3751880150584993549UL, 3481, 59, 118, 18446744073709551498UL, 0, 0);

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void factorial_verifyslow(	__global ulong4 * g_verify,
											const uint startN ){
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint gs = get_global_size(0);
	__local ulong total[256];
	ulong thread_total = prime.s3;
	bool first_iter = true;

	for(uint currN = 2 + gid; currN <= startN; currN += gs){

		ulong n = m_mul( currN, prime.s2, prime.s0, prime.s1);	// convert N to montgomery form

		if(first_iter){
			first_iter = false;
			thread_total = n;
		}
		else{
			thread_total = m_mul( thread_total, n, prime.s0, prime.s1);
		}
	}

	total[lid] = thread_total;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid] = m_mul( total[lid], total[lid+s], prime.s0, prime.s1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_verify[get_group_id(0)].s0 = total[0];
	}

}


__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void primorial_verifyslow(	__global ulong4 * g_verify,
											__global uint * g_primes,
											const uint primesize ){
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint gs = get_global_size(0);
	__local ulong total[256];
	ulong thread_total = prime.s3;
	bool first_iter = true;

	for(uint i=gid; i<primesize; i+=gs){
		ulong n = m_mul( g_primes[i], prime.s2, prime.s0, prime.s1);
		if(first_iter){
			first_iter = false;
			thread_total = n;
		}
		else{
			thread_total = m_mul( thread_total, n, prime.s0, prime.s1);
		}
	}

	total[lid] = thread_total;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid] = m_mul( total[lid], total[lid+s], prime.s0, prime.s1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_verify[get_group_id(0)].s0 = total[0];
	}

}


__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void compositorial_verifyslow(	__global ulong4 * g_verify,
												__global uint * g_primes,
												const uint primesize,
												const uint nmax ){
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint gs = get_global_size(0);
	__local ulong total[256];
	ulong thread_total = prime.s3;
	bool first_iter = true;

	uint k=0;
	for(uint i=gid+2; i<nmax; i+=gs){
		for(; k<primesize && g_primes[k] < i; ++k);
		if(g_primes[k] == i) continue;
		ulong n = m_mul( i, prime.s2, prime.s0, prime.s1);
		if(first_iter){
			first_iter = false;
			thread_total = n;
		}
		else{
			thread_total = m_mul( thread_total, n, prime.s0, prime.s1);
		}

	}

	total[lid] = thread_total;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid] = m_mul( total[lid], total[lid+s], prime.s0, prime.s1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_verify[get_group_id(0)].s0 = total[0];
	}

}






