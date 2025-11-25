/* 

	verifyresult.cl - Bryan Little 4/2025, montgomery arithmetic by Yves Gallot
	
	verify power table final reduction and result

*/

// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo
__constant ulong8 prime = (ulong8)(18446744073709551557UL, 3751880150584993549UL, 3481, 59, 118, 18446744073709551498UL, 0, 0);

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void verifyreduce(	__global ulong4 * g_verify,
										const uint num_totals ){
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	__local ulong2 total[256];

	if(gid < num_totals){
		total[lid].s0 = g_verify[gid].s0;
		total[lid].s1 = g_verify[gid].s1;
	}
	else{
		total[lid].s0 = prime.s3;
		total[lid].s1 = prime.s3;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid].s0 = m_mul( total[lid].s0, total[lid+s].s0, prime.s0, prime.s1);
			total[lid].s1 = m_mul( total[lid].s1, total[lid+s].s1, prime.s0, prime.s1);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		g_verify[get_group_id(0)].s2 = total[0].s0;
		g_verify[get_group_id(0)].s3 = total[0].s1;
	}

}

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void verifyresult(	__global ulong4 * g_verify,
										__global uint *g_primecount,
										const uint num_totals ){
	const uint lid = get_local_id(0);
	__local ulong2 total[256];

	if(lid < num_totals){					// needs to be <=256
		total[lid].s0 = g_verify[lid].s2;
		total[lid].s1 = g_verify[lid].s3;
	}
	else{
		total[lid].s0 = prime.s3;
		total[lid].s1 = prime.s3;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid].s0 = m_mul( total[lid].s0, total[lid+s].s0, prime.s0, prime.s1);
			total[lid].s1 = m_mul( total[lid].s1, total[lid+s].s1, prime.s0, prime.s1);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		// factorial calculated with power table should match iterative algorithm
		if( total[0].s0 != total[0].s1 || (total[0].s0 == 0 && total[0].s1 == 0) ){
			g_primecount[3] = 1;
		}
	}
}





