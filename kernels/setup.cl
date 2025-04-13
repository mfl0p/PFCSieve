/*

	setup.cl - Bryan Little 4/2025, montgomery arithmetic by Yves Gallot
	
	generates nmin! mod P using power or product tables

	The CPU will run this kernel in many small chunks to limit kernel runtime.

	Setup and iterate kernels are the main compute intensive kernels.

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

__kernel void factorial_setup(	__global ulong8 * g_prime,
				__global uint * g_primecount,
			 	__global ulong * g_smallprimeprod,
				const uint start,
				const uint end,
				__global uint2 * g_smallpowers,
				const uint startN) {

	const uint gid = get_global_id(0);

	if(gid >= g_primecount[0]) return;

	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of startN! mod P, .s7=startN in montgomery form
	ulong8 prime = g_prime[gid];
	uint i = start;

	if(!start){
		++i;
		// setup r2 and montgomery form of startN
		prime.s2 = add(prime.s4, prime.s4, prime.s0);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);	// 4^{2^5} = 2^64
		g_prime[gid].s2 = prime.s2;
		g_prime[gid].s7 = m_mul(startN, prime.s2, prime.s0, prime.s1);
		// first iteration, base prime = 2
		// .s0=exp, .s1=curBit
		uint2 p = g_smallpowers[0];
		// left to right powmod
		ulong a = prime.s4;
		while( p.s1 ){
			a = m_mul(a, a, prime.s0, prime.s1);
			if(p.s0 & p.s1){
				a = add(a, a, prime.s0);		// base 2 we can add
			}
			p.s1 >>= 1;
		}
		prime.s6 = a;
	}
	for(; i<end; ++i){
		// remaining iterations, starting at prime = 3
		ulong sm_prime = g_smallprimeprod[i];
		// .s0=exp, .s1=curBit
		uint2 p = g_smallpowers[i];
		const ulong base = m_mul(sm_prime, prime.s2, prime.s0, prime.s1);
		ulong primepow;
		if(p.s0 == 1){
			primepow = base;
		}
		else{
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
		prime.s6 = m_mul(prime.s6, primepow, prime.s0, prime.s1);
	}

	// done with power table, store to global
	// residue is equal to startN! mod P
	g_prime[gid].s6 = prime.s6;
}


__kernel void primorial_setup(	__global ulong8 * g_prime,
				__global uint * g_primecount,
				__global ulong * g_smallprimeprod,
				const uint start,
				const uint end) {

	const uint gid = get_global_id(0);

	if(gid >= g_primecount[0]) return;

	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of start# mod P
	ulong8 prime = g_prime[gid];

	if(!start){
		prime.s2 = add(prime.s4, prime.s4, prime.s0);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);		// 4^{2^5} = 2^64
		g_prime[gid].s2 = prime.s2;
	}

	for(uint i=start; i<end; ++i){
		ulong p = g_smallprimeprod[i];
		ulong montprimprod = m_mul(p, prime.s2, prime.s0, prime.s1);
		if(i){
			prime.s6 = m_mul(prime.s6, montprimprod, prime.s0, prime.s1);
		}
		else{
			prime.s6 = montprimprod;
		}
	}

	// done with initial primorial, store to global
	// residue is equal to start# mod P
	g_prime[gid].s6 = prime.s6;
}


__kernel void compositorial_setup(	__global ulong8 * g_prime,
					__global uint * g_primecount,
				 	__global ulong * g_smallcompprod,
					const uint start,
					const uint end,
					const uint startN) {

	const uint gid = get_global_id(0);

	if(gid >= g_primecount[0]) return;

	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of start!/# mod P, .s7=startN in montgomery form
	ulong8 prime = g_prime[gid];

	if(!start){
		prime.s2 = add(prime.s4, prime.s4, prime.s0);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);		// 4^{2^5} = 2^64
		g_prime[gid].s2 = prime.s2;
		g_prime[gid].s7 = m_mul(startN, prime.s2, prime.s0, prime.s1);
	}

	for(uint i=start; i<end; ++i){
		ulong p = g_smallcompprod[i];
		ulong montcompprod = m_mul(p, prime.s2, prime.s0, prime.s1);
		if(i){
			prime.s6 = m_mul(prime.s6, montcompprod, prime.s0, prime.s1);
		}
		else{
			prime.s6 = montcompprod;
		}
	}

	// done with initial primorial, store to global
	// residue is equal to start!/# mod P
	g_prime[gid].s6 = prime.s6;
}


__kernel void combined_setup(	__global ulong8 * g_prime,
				__global uint * g_primecount,
			 	__global ulong * g_smallprimeprod,
				const uint start,
				const uint end,
				__global uint2 * g_smallpowers,
				const uint startN,
			 	__global ulong * g_smallcompprod,
				const uint f_end,
				const uint c_end) {

	const uint gid = get_global_id(0);

	if(gid >= g_primecount[0]) return;

	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of startN!, .s7=startN in montgomery form
	ulong8 prime = g_prime[gid];
	uint i = start;

	// first iteration of kernel
	if(!start){
		++i;
		// setup r2 and montgomery form of startN
		// after r2 setup, .s4 is now used for compositorial residue
		prime.s2 = add(prime.s4, prime.s4, prime.s0);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);
		prime.s2 = m_mul(prime.s2, prime.s2, prime.s0, prime.s1);	// 4^{2^5} = 2^64
		g_prime[gid].s2 = prime.s2;
		g_prime[gid].s7 = m_mul(startN, prime.s2, prime.s0, prime.s1);
		// first iteration of factorial power table, base prime = 2
		// .s0=exp, .s1=curBit
		uint2 power = g_smallpowers[0];
		// left to right powmod
		ulong a = prime.s4;
		while( power.s1 ){
			a = m_mul(a, a, prime.s0, prime.s1);
			if(power.s0 & power.s1){
				a = add(a, a, prime.s0);		// base 2 we can add
			}
			power.s1 >>= 1;
		}
		prime.s6 = a;
		// first iteration of compositorial product table
		ulong prod = g_smallcompprod[0];
		prime.s4 = m_mul(prod, prime.s2, prime.s0, prime.s1);
	}

	// remaining iterations of factorial power table, starting at prime = 3
	uint loop_end = end > f_end ? f_end : end;
	for(uint k=i; k<loop_end; ++k){
		ulong sm_prime = g_smallprimeprod[k];
		// .s0=exp, .s1=curBit
		uint2 power = g_smallpowers[k];
		const ulong base = m_mul(sm_prime, prime.s2, prime.s0, prime.s1);
		ulong primepow;
		if(power.s0 == 1){
			primepow = base;
		}
		else{
			ulong a = base;
			while( power.s1 ){
				a = m_mul(a, a, prime.s0, prime.s1);
				if(power.s0 & power.s1){
					a = m_mul(a, base, prime.s0, prime.s1);
				}
				power.s1 >>= 1;
			}
			primepow = a;
		}
		prime.s6 = m_mul(prime.s6, primepow, prime.s0, prime.s1);
	}

	loop_end = end > c_end ? c_end : end;
	for(uint k=i; k<loop_end; ++k){
		ulong prod = g_smallcompprod[k];
		ulong montcompprod = m_mul(prod, prime.s2, prime.s0, prime.s1);
		prime.s4 = m_mul(prime.s4, montcompprod, prime.s0, prime.s1);
	}

	// done with power/product tables, store to global
	g_prime[gid].s4 = prime.s4;
	g_prime[gid].s6 = prime.s6;

}






