/*

	verify.cl - Bryan Little 11/2025, montgomery arithmetic by Yves Gallot
	
	factorial function calculates the starting N factorial quickly using prime powers
	primorial function calculates the primorial of the ending N using the initial primorial table and the iteration table

*/

// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo
__constant ulong8 prime = (ulong8)(18446744073709551557UL, 3751880150584993549UL, 3481, 59, 118, 18446744073709551498UL, 0, 0);

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void factorial_verify(	__global ulong * g_smallprimes,
											__global uint2 * g_smallpowers,
											__global ulong4 * g_verify,
											const uint smallcount) {

	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint gs = get_global_size(0);
	__local ulong total[256];
	bool first_iter = true;
	ulong thread_total = prime.s3;

	for(uint position = gid; position < smallcount; position+=gs){
		ulong sm_prime = g_smallprimes[position];
		// .s0=exp, .s1=curBit
		uint2 p = g_smallpowers[position];
		const ulong base = m_mul(sm_prime, prime.s2, prime.s0, prime.s1);
		ulong primepow;
		if(p.s0 == 1){
			primepow = base;
		}
		else{
			// left to right powmod
			ulong a = base;
			while( p.s1 ){
				a = m_mul(a, a, prime.s0, prime.s1);
				if(p.s0 & p.s1){
					a = m_mul(a, base, prime.s0, prime.s1);
				}
				p.s1 >>= 1;
			}
			primepow = a;
		}
		if(first_iter){
			first_iter = false;
			thread_total = primepow;
		}
		else{
			thread_total = m_mul(thread_total, primepow, prime.s0, prime.s1);
		}
	}

	total[lid] = thread_total;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid] = m_mul(total[lid], total[lid+s], prime.s0, prime.s1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_verify[get_group_id(0)].s1 = total[0];
	}


}


__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void primorial_verify(	__global ulong4 * g_verify,
											__global ulong * g_products,
											__global uint * g_primes,
											const uint prodsize,
											const uint itersize ){
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint gs = get_global_size(0);
	__local ulong total[256];
	ulong thread_total = prime.s3;
	bool first_iter = true;

	for(uint i=gid; i<prodsize; i+=gs){
		ulong n = m_mul( g_products[i], prime.s2, prime.s0, prime.s1);
		if(first_iter){
			first_iter = false;
			thread_total = n;
		}
		else{
			thread_total = m_mul( thread_total, n, prime.s0, prime.s1);
		}
	}

	for(uint i=gid; i<itersize; i+=gs){
		ulong n = m_mul( g_primes[i], prime.s2, prime.s0, prime.s1);
		thread_total = m_mul( thread_total, n, prime.s0, prime.s1);
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
		g_verify[get_group_id(0)].s1 = total[0];
	}

}




