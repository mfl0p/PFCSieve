/* 

	iterate.cl - Bryan Little 11/2025, montgomery arithmetic by Yves Gallot
	
	iterate from nmin to nmax-1 mod P

	The CPU will run this kernel in many small chunks to limit kernel runtime.

	Iterate and setup kernels are the main compute intensive kernels.
	
*/

#define FACTORIAL 0
#define PRIMORIAL 1
#define COMPOSITORIAL 2

typedef struct {
	ulong p;
	int nc;
	int type;
}factor;

__kernel void primorial_iterate(__global ulong8 * g_prime,
				__global uint * g_primecount,
				__global factor * g_factor,
				const uint start,
				const uint end,
				__global uint * g_smallprimes ){

	const uint gid = get_global_id(0);

	if(gid >= g_primecount[0]) return;

	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue, .s7=N in montgomery form
	ulong8 prime = g_prime[gid];

	for(uint j=start; j<end; ++j){
		uint p = g_smallprimes[j];
		ulong montprime = m_mul(p, prime.s2, prime.s0, prime.s1);
		prime.s6 = m_mul(prime.s6, montprime, prime.s0, prime.s1);
		if(prime.s6 == prime.s3 || prime.s6 == prime.s5){
			uint i = atomic_inc(&g_primecount[2]);
			factor fac = {prime.s0, (prime.s6 == prime.s3) ? -((int)p) : (int)p, PRIMORIAL};
			g_factor[i] = fac;
		}
	}

	// store final residue
	g_prime[gid].s6 = prime.s6;
}


__kernel void fc_iterate(	__global ulong8 * g_prime,
				__global uint * g_primecount,
				__global factor * g_factor,
				const uint startN,
				const uint endN
#ifdef COMP
				, __global uint * g_smallprimes,
				const uint primeposition
#endif
				){

	const uint gid = get_global_id(0);

	if(gid >= g_primecount[0]) return;

	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=residue of start!/#, .s5=nmo, .s6=residue of start!, .s7=N in montgomery form
	ulong8 prime = g_prime[gid];
#ifdef COMP
	uint ppos = primeposition;
	uint nextprime = g_smallprimes[ppos];
	if(!prime.s0) return;
#endif

	for(uint currN = startN; currN < endN; ++currN){
		prime.s7 = add(prime.s7, prime.s3, prime.s0);
#ifdef FACT
		prime.s6 = m_mul(prime.s6, prime.s7, prime.s0, prime.s1);
		if(prime.s6 == prime.s3 || prime.s6 == prime.s5){
			uint i = atomic_inc(&g_primecount[2]);		// found factorial factor
			factor fac = {prime.s0, (prime.s6 == prime.s3) ? -((int)currN) : (int)currN, FACTORIAL};
			g_factor[i] = fac;
		}
#endif
#ifdef COMP
		if(currN == nextprime){
			nextprime = g_smallprimes[++ppos];
			continue;
		}
		prime.s4 = m_mul(prime.s4, prime.s7, prime.s0, prime.s1);
		if(prime.s4 == prime.s3 || prime.s4 == prime.s5){
			uint i = atomic_inc(&g_primecount[2]);		// found compositorial factor
			factor fac = {prime.s0, (prime.s4 == prime.s3) ? -((int)currN) : (int)currN, COMPOSITORIAL};
			g_factor[i] = fac;
		}
#endif
	}

	// store final residues and n
#ifdef COMP
	g_prime[gid].s4 = prime.s4;
#endif
#ifdef FACT
	g_prime[gid].s6 = prime.s6;
#endif
	g_prime[gid].s7 = prime.s7;

}



