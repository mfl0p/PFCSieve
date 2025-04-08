
// cl_sieve.h

typedef struct {
	cl_ulong p;
	cl_int nc;
}factor;

typedef struct {
	uint64_t pmin, pmax, p, checksum, primecount, factorcount, last_trickle, state_sum;
	uint32_t nmin, nmax;
	bool factorial, primorial, compositorial;
}workStatus;

typedef struct {
	uint64_t maxmalloc;
	uint32_t computeunits, nstep, sstep, smcount, numresults, threadcount, range, psize, numgroups, nlimit;
	bool test, compute, write_state_a_next;
}searchData;

typedef struct {
	cl_mem d_factor;
	cl_mem d_sum;
	cl_mem d_primes;
	cl_mem d_primecount;
	cl_mem d_smallprimes;
	cl_mem d_powers;
	cl_mem d_products;
	sclSoft check, iterate, clearn, clearresult, setup, getsegprimes, addsmallprimes, verifyslow, verify, verifyreduce, verifyresult;
}progData;

void cl_sieve( sclHard hardware, workStatus & st, searchData & sd );

void run_test( sclHard hardware, workStatus & st, searchData & sd );
