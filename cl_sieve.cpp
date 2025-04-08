/*
	PFCSieve
	Bryan Little, April 2025
	
	with contributions by Yves Gallot, Mark Rodenkirch, and Kim Walisch

	Required minimum OpenCL version is 1.1
	CL_TARGET_OPENCL_VERSION 110 in simpleCL.h

	Search limits:  P up to 2^64 and N up to 2^31

	Using OpenMP for multithreaded factor verification.

*/

#include <unistd.h>
#include <cinttypes>
#include <math.h>
#include <omp.h>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"

#include "check.h"
#include "clearn.h"
#include "clearresult.h"
#include "getsegprimes.h"
#include "addsmallprimes.h"
#include "setup.h"
#include "iterate.h"
#include "verifyslow.h"
#include "verify.h"
#include "verifyreduce.h"
#include "verifyresult.h"
#include "primcheck.h"
#include "primsetup.h"
#include "primiterate.h"
#include "primverifyslow.h"
#include "primverify.h"
#include "compsetup.h"
#include "compiterate.h"
#include "compverifyslow.h"
#include "compverify.h"

#include "primesieve.h"
#include "putil.h"
#include "cl_sieve.h"
#include "verifyprime.h"

#define RESULTS_FILENAME "factors.txt"
#define STATE_FILENAME_A "stateA.ckp"
#define STATE_FILENAME_B "stateB.ckp"



void handle_trickle_up(workStatus & st){
	if(boinc_is_standalone()) return;
	uint64_t now = (uint64_t)time(NULL);
	if( (now-st.last_trickle) > 86400 ){	// Once per day
		st.last_trickle = now;
		double progress = boinc_get_fraction_done();
		double cpu;
		boinc_wu_cpu_time(cpu);
		APP_INIT_DATA init_data;
		boinc_get_init_data(init_data);
		double run = boinc_elapsed_time() + init_data.starting_elapsed_time;
		char msg[512];
		sprintf(msg, "<trickle_up>\n"
			    "   <progress>%lf</progress>\n"
			    "   <cputime>%lf</cputime>\n"
			    "   <runtime>%lf</runtime>\n"
			    "</trickle_up>\n",
			     progress, cpu, run  );
		char variety[64];
		sprintf(variety, "pfsieve_progress");
		boinc_send_trickle_up(variety, msg);
	}
}


FILE *my_fopen(const char * filename, const char * mode){
	char resolved_name[512];
	boinc_resolve_filename(filename,resolved_name,sizeof(resolved_name));
	return boinc_fopen(resolved_name,mode);
}


void cleanup( progData & pd, searchData & sd, workStatus & st ){
	sclReleaseMemObject(pd.d_factor);
	sclReleaseMemObject(pd.d_sum);
	sclReleaseMemObject(pd.d_primes);
	sclReleaseMemObject(pd.d_primecount);
	sclReleaseMemObject(pd.d_products);
	sclReleaseClSoft(pd.check);
	sclReleaseClSoft(pd.clearn);
	sclReleaseClSoft(pd.clearresult);
        sclReleaseClSoft(pd.iterate);
        sclReleaseClSoft(pd.setup);
        sclReleaseClSoft(pd.getsegprimes);
        sclReleaseClSoft(pd.addsmallprimes);
	sclReleaseClSoft(pd.verifyslow);
	sclReleaseClSoft(pd.verify);
	sclReleaseClSoft(pd.verifyreduce);
	sclReleaseClSoft(pd.verifyresult);
	if(st.factorial){
		sclReleaseMemObject(pd.d_powers);
	}
	else{
		sclReleaseMemObject(pd.d_smallprimes);
	}
}


// using fast binary checkpoint files with checksum calculation
void write_state( workStatus & st, searchData & sd ){

	FILE * out;

	st.state_sum = st.pmin+st.pmax+st.p+st.checksum+st.primecount+st.factorcount+st.last_trickle+st.nmin+st.nmax;

        if (sd.write_state_a_next){
		if ((out = my_fopen(STATE_FILENAME_A,"wb")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_A);
	}
	else{
                if ((out = my_fopen(STATE_FILENAME_B,"wb")) == NULL)
                        fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_B);
        }

	if(out != NULL){

		if( fwrite(&st, sizeof(workStatus), 1, out) != 1 ){
			fprintf(stderr,"Cannot write checkpoint to file. Continuing...\n");
			// Attempt to close, even though we failed to write
			fclose(out);
		}
		else{
			// If state file is closed OK, write to the other state file
			// next time around
			if (fclose(out) == 0) 
				sd.write_state_a_next = !sd.write_state_a_next; 
		}
	}
}


int read_state( workStatus & st, searchData & sd ){

	FILE * in;
	bool good_state_a = true;
	bool good_state_b = true;
	workStatus stat_a, stat_b;

        // Attempt to read state file A
	if ((in = my_fopen(STATE_FILENAME_A,"rb")) == NULL){
		good_state_a = false;
        }
	else{
		if( fread(&stat_a, sizeof(workStatus), 1, in) != 1 ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_A);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else if(stat_a.pmin != st.pmin || stat_a.pmax != st.pmax || stat_a.nmin != st.nmin || stat_a.nmax != st.nmax
			|| stat_a.factorial != st.factorial || stat_a.primorial != st.primorial || stat_a.compositorial != st.compositorial){
			fprintf(stderr,"Invalid checkpoint file %s !!!\n",STATE_FILENAME_A);
			printf("Invalid checkpoint file %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else{
			uint64_t state_sum = stat_a.pmin+stat_a.pmax+stat_a.p+stat_a.checksum+stat_a.primecount+stat_a.factorcount
						+stat_a.last_trickle+stat_a.nmin+stat_a.nmax;
			if(state_sum != stat_a.state_sum){
				fprintf(stderr,"Checksum error in %s !!!\n",STATE_FILENAME_A);
				printf("Checksum error in %s !!!\n",STATE_FILENAME_A);
				good_state_a = false;
			}
		}
		fclose(in);
	}

        // Attempt to read state file B
	if ((in = my_fopen(STATE_FILENAME_B,"rb")) == NULL){
		good_state_b = false;
        }
	else{
		if( fread(&stat_b, sizeof(workStatus), 1, in) != 1 ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_B);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else if(stat_b.pmin != st.pmin || stat_b.pmax != st.pmax || stat_b.nmin != st.nmin || stat_b.nmax != st.nmax
			|| stat_b.factorial != st.factorial || stat_b.primorial != st.primorial || stat_b.compositorial != st.compositorial){
			fprintf(stderr,"Invalid checkpoint file %s !!!\n",STATE_FILENAME_B);
			printf("Invalid checkpoint file %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else{
			uint64_t state_sum = stat_b.pmin+stat_b.pmax+stat_b.p+stat_b.checksum+stat_b.primecount+stat_b.factorcount
						+stat_b.last_trickle+stat_b.nmin+stat_b.nmax;
			if(state_sum != stat_b.state_sum){
				fprintf(stderr,"Checksum error in %s !!!\n",STATE_FILENAME_B);
				printf("Checksum error in %s !!!\n",STATE_FILENAME_B);
				good_state_b = false;
			}
		}
		fclose(in);
	}

        // If both state files are OK, check which is the most recent
	if (good_state_a && good_state_b)
	{
		if (stat_a.p > stat_b.p)
			good_state_b = false;
		else
			good_state_a = false;
	}

        // Use data from the most recent state file
	if (good_state_a && !good_state_b)
	{
		memcpy(&st, &stat_a, sizeof(workStatus));
		sd.write_state_a_next = false;
		if(boinc_is_standalone()){
			printf("Resuming from checkpoint in %s\n",STATE_FILENAME_A);
		}
		return 1;
	}
        if (good_state_b && !good_state_a)
        {
		memcpy(&st, &stat_b, sizeof(workStatus));
		sd.write_state_a_next = true;
		if(boinc_is_standalone()){
			printf("Resuming from checkpoint in %s\n",STATE_FILENAME_B);
		}
		return 1;
        }

	// If we got here, neither state file was good
	return 0;
}


void checkpoint( workStatus & st, searchData & sd ){
	handle_trickle_up( st );
	write_state( st, sd );
	if(boinc_is_standalone()){
		printf("Checkpoint, current p: %" PRIu64 "\n", st.p);
	}
	boinc_checkpoint_completed();
}


// sleep CPU thread while waiting on the specified event to complete in the command queue
// using critical sections to prevent BOINC from shutting down the program while kernels are running on the GPU
void waitOnEvent(sclHard hardware, cl_event event){

	cl_int err;
	cl_int info;
#ifdef _WIN32
#else
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
       	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
	       	}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(event);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
		       	}

			boinc_end_critical_section();

			return;
		}
	}
}


// queue a marker and sleep CPU thread until marker has been reached in the command queue
void sleepCPU(sclHard hardware){

	cl_event kernelsDone;
	cl_int err;
	cl_int info;
#ifdef _WIN32
#else
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	// OpenCL v2.0
/*
	err = clEnqueueMarkerWithWaitList( hardware.queue, 0, NULL, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarkerWithWaitList\n");
		fprintf(stderr, "ERROR: clEnqueueMarkerWithWaitList\n");
		sclPrintErrorFlags(err); 
	}
*/
	err = clEnqueueMarker( hardware.queue, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarker\n");
		fprintf(stderr, "ERROR: clEnqueueMarker\n");
		sclPrintErrorFlags(err); 
	}

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
       	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(kernelsDone, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
	       	}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(kernelsDone);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
		       	}

			boinc_end_critical_section();

			return;
		}
	}
}



// find mod 30 wheel index based on starting N
// this is used by gpu threads to iterate over the number line
void findWheelOffset(uint64_t & start, int32_t & index){

	int32_t wheel[8] = {4, 2, 4, 2, 4, 6, 2, 6};
	int32_t idx = -1;

	// find starting number using mod 6 wheel
	// N=(k*6)-1, N=(k*6)+1 ...
	// where k, k+1, k+2 ...
	uint64_t k = start / 6;
	int32_t i = 1;
	uint64_t N = (k * 6)-1;


	while( N < start || N % 5 == 0 ){
		if(i){
			i = 0;
			N += 2;
		}
		else{
			i = 1;
			N += 4;
		}
	}

	start = N;

	// find mod 30 wheel index by iterating with a mod 6 wheel until finding N divisible by 5
	// forward to find index
	while(idx < 0){

		if(i){
			N += 2;
			i = 0;
			if(N % 5 == 0){
				N -= 2;
				idx = 5;
			}

		}
		else{
			N += 4;
			i = 1;
			if(N % 5 == 0){
				N -= 4;
				idx = 7;
			}
		}
	}

	// reverse to find starting index
	while(N != start){
		--idx;
		if(idx < 0)idx=7;
		N -= wheel[idx];
	}


	index = idx;

}


int factorcompare(const void *a, const void *b) {
  	factor *factA = (factor *)a;
	factor *factB = (factor *)b;
	if(factB->p < factA->p){
		return 1;
	}
	else if(factB->p == factA->p){
		int32_t nA = (factA->nc < 0) ? -factA->nc : factA->nc;
		int32_t nB = (factB->nc < 0) ? -factB->nc : factB->nc;
		if(nB < nA){
			return 1;
		}
	}
	return -1;
}


void getResults( progData & pd, workStatus & st, searchData & sd, sclHard hardware, uint64_t * h_checksum, uint32_t * h_primecount, uint32_t * verifylist, size_t verifylistsize ){
	// copy checksum and total prime count to host memory, non-blocking
	sclReadNB(hardware, sd.numgroups*sizeof(uint64_t), pd.d_sum, h_checksum);
	// copy prime count to host memory, blocking
	sclRead(hardware, 6*sizeof(uint32_t), pd.d_primecount, h_primecount);
	// index 0 is the gpu's total prime count
	st.primecount += h_checksum[0];
	// sum blocks
	for(uint32_t i=1; i<sd.numgroups; ++i){
		st.checksum += h_checksum[i];
	}
	// largest kernel prime count.  used to check array bounds
	if(h_primecount[1] > sd.psize){
		fprintf(stderr,"error: gpu prime array overflow\n");
		printf("error: gpu prime array overflow\n");
		exit(EXIT_FAILURE);
	}
	// flag set if there is a gpu overflow error
	if(h_primecount[4] == 1){
		fprintf(stderr,"error: getsegprimes kernel local memory overflow\n");
		printf("error: getsegprimes kernel local memory overflow\n");
		exit(EXIT_FAILURE);
	}
	// flag set if there is a gpu validation failure
	if(h_primecount[5] == 1){
		fprintf(stderr,"error: gpu validation failure\n");
		printf("error: gpu validation failure\n");
		exit(EXIT_FAILURE);
	}
	uint32_t numfactors = h_primecount[2];
	if(numfactors > 0){
		if(boinc_is_standalone()){
			printf("processing %u factors on CPU\n", numfactors);
		}
		if(numfactors > sd.numresults){
			fprintf(stderr,"Error: number of results (%u) overflowed array.\n", numfactors);
			exit(EXIT_FAILURE);
		}
		factor * h_factor = (factor *)malloc(numfactors * sizeof(factor));
		if( h_factor == NULL ){
			fprintf(stderr,"malloc error: h_factor\n");
			exit(EXIT_FAILURE);
		}
		// copy factors to host memory, blocking
		sclRead(hardware, numfactors * sizeof(factor), pd.d_factor, h_factor);
		// sort results by prime size if needed
		if(numfactors > 1){
			if(boinc_is_standalone()){
				printf("sorting factors\n");
			}
			qsort(h_factor, numfactors, sizeof(factor), factorcompare);
		}
		// verify all factors on CPU using slow test
		if(boinc_is_standalone()){
			printf("Verifying factors on CPU...\n");
		}

		double last = 0.0;
		uint32_t tested = 0;
		#pragma omp parallel for
		for(uint32_t i=0; i<numfactors; ++i){
			uint64_t fp = h_factor[i].p;
			uint32_t fn = (h_factor[i].nc < 0) ? -h_factor[i].nc : h_factor[i].nc; 
			int32_t fc = (h_factor[i].nc < 0) ? -1 : 1;
			if( !verify( fp, fn, fc, st.factorial, st.primorial, st.compositorial, verifylist, verifylistsize ) ){
				if(st.factorial){
					fprintf(stderr,"CPU factor verification failed!  %" PRIu64 " is not a factor of %u!%+d\n", fp, fn, fc);
					printf("\nCPU factor verification failed!  %" PRIu64 " is not a factor of %u!%+d\n", fp, fn, fc);
				}
				else if(st.primorial){
					fprintf(stderr,"CPU factor verification failed!  %" PRIu64 " is not a factor of %u#%+d\n", fp, fn, fc);
					printf("\nCPU factor verification failed!  %" PRIu64 " is not a factor of %u#%+d\n", fp, fn, fc);
				}
				else if(st.compositorial){
					fprintf(stderr,"CPU factor verification failed!  %" PRIu64 " is not a factor of %u!/#%+d\n", fp, fn, fc);
					printf("\nCPU factor verification failed!  %" PRIu64 " is not a factor of %u!/#%+d\n", fp, fn, fc);
				}
				exit(EXIT_FAILURE);
			}
			if(boinc_is_standalone()){
				#pragma omp atomic
				++tested;
				double done = (double)(tested+1) / (double)numfactors * 100.0;
				if(done > last+0.1){
					last = done;
					printf("\r%.1f%%     ",done);
					fflush(stdout);
				}
			}
		}

		fprintf(stderr,"Verified %u factors.\n", numfactors);
		if(boinc_is_standalone()){
			printf("\rVerified %u factors.\n", numfactors);
		}
		// write factors to file
		FILE * resfile = my_fopen(RESULTS_FILENAME,"a");
		if( resfile == NULL ){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
		if(boinc_is_standalone()){
			printf("writing factors to %s\n", RESULTS_FILENAME);
		}
		uint64_t lastgoodp = 0;
		for(uint32_t i=0; i<numfactors; ++i){
			uint64_t fp = h_factor[i].p;
			uint32_t fn = (h_factor[i].nc < 0) ? -h_factor[i].nc : h_factor[i].nc; 
			int32_t fc = (h_factor[i].nc < 0) ? -1 : 1;
			if( fp == lastgoodp || isPrime(fp) ){	// gpu generates 2-PRPs, we only want prime factors
				lastgoodp = fp;
				++st.factorcount;
				if(st.factorial){
					if( fprintf( resfile, "%" PRIu64 " | %u!%+d\n",fp,fn,fc) < 0 ){
						fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
						exit(EXIT_FAILURE);
					}
				}
				else if(st.primorial){
					if( fprintf( resfile, "%" PRIu64 " | %u#%+d\n",fp,fn,fc) < 0 ){
						fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
						exit(EXIT_FAILURE);
					}
				}
				else if(st.compositorial){
					if( fprintf( resfile, "%" PRIu64 " | %u!/#%+d\n",fp,fn,fc) < 0 ){
						fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
						exit(EXIT_FAILURE);
					}
				}
				// add the factor to checksum
				st.checksum += fn + fc;
			}
			else{
				fprintf(stderr,"discarded 2-PRP factor %" PRIu64 "\n", fp);
				printf("discarded 2-PRP factor %" PRIu64 "\n", fp);
			}	
		}
		fclose(resfile);
		free(h_factor);
	}
}


void setupSearch(workStatus & st, searchData & sd){

	st.p = st.pmin;

	int z=0;
	if(st.factorial)++z;
	if(st.primorial)++z;
	if(st.compositorial)++z;
	if(!z){
		printf("\n-! or -# or -c argument is required\nuse -h for help\n");
		fprintf(stderr, "-! or -# or -c argument is required\nuse -h for help\n");
		exit(EXIT_FAILURE);
	}
	else if(z>1){
		printf("\nSelect only one test type!\nuse -h for help\n");
		fprintf(stderr, "Select only one test type!\nuse -h for help\n");
		exit(EXIT_FAILURE);
	}

	if(st.pmin == 0 || st.pmax == 0){
		printf("\n-p and -P arguments are required\nuse -h for help\n");
		fprintf(stderr, "-p and -P arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if(st.nmin == 0 || st.nmax == 0){
		printf("\n-n and -N arguments are required\nuse -h for help\n");
		fprintf(stderr, "-n and -N arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if (st.nmin > st.nmax){
		printf("nmin <= nmax is required\nuse -h for help\n");
		fprintf(stderr, "nmin <= nmax is required\n");
		exit(EXIT_FAILURE);
	}

	if (st.pmin > st.pmax){
		printf("pmin <= pmax is required\nuse -h for help\n");
		fprintf(stderr, "pmin <= pmax is required\n");
		exit(EXIT_FAILURE);
	}

	if (st.pmin < st.nmin && (st.factorial || st.primorial) ){
		printf("for factorial and primorial pmin must be >= nmin, there are no factors when p <= nmin\nuse -h for help\n");
		fprintf(stderr, "for factorial and primorial pmin must be >= nmin, there are no factors when p <= nmin\n");
		exit(EXIT_FAILURE);
	}

	// increase result buffer at low P range
	// it's still possible to overflow this with a fast GPU and large search range
	if(st.pmin < 0xFFFFFFFF){
		sd.numresults = 30000000;
	}

	fprintf(stderr, "Starting sieve at p: %" PRIu64 " n: %u\nStopping sieve at P: %" PRIu64 " N: %u\n", st.pmin, st.nmin, st.pmax, st.nmax);
	if(boinc_is_standalone()){
		printf("Starting sieve at p: %" PRIu64 " n: %u\nStopping sieve at P: %" PRIu64 " N: %u\n", st.pmin, st.nmin, st.pmax, st.nmax);
	}

	// setup and iterate kernel size
	if(sd.compute){
		sd.sstep = 25 * sd.computeunits;
		sd.nstep = 300 * sd.computeunits;
	}
	else{
		sd.sstep = 9 * sd.computeunits;
		sd.nstep = 60 * sd.computeunits;
	}


}



void profileGPU(progData & pd, workStatus & st, searchData & sd, sclHard hardware){

	// calculate approximate chunk size based on gpu's compute units
	cl_int err = 0;

	uint64_t calc_range = sd.computeunits * (uint64_t)350000;

	// limit kernel global size
	if(calc_range > 4294900000){
		calc_range = 4294900000;
	}

	uint64_t start = st.p;
	uint64_t stop = start + calc_range;

	// check overflow at 2^64
	if(stop < start){
		stop = 0xFFFFFFFFFFFFFFFF;
		calc_range = stop - start;
	}

	sclSetGlobalSize( pd.getsegprimes, (calc_range/60)+1 );

	// get a count of primes in the gpu worksize
	uint64_t range_primes = (stop / log(stop)) - (start / log(start));

	// calculate prime array size based on result
	uint64_t mem_size = (uint64_t)(1.5 * (double)range_primes);

	// kernels use uint for global id
	if(mem_size > UINT32_MAX){
		fprintf(stderr, "ERROR: mem_size too large.\n");
                printf( "ERROR: mem_size too large.\n" );
		exit(EXIT_FAILURE);
	}

	// allocate temporary gpu prime array for profiling
	cl_mem d_profileprime = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, mem_size*sizeof(cl_ulong8), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
	        printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	int32_t wheelidx;
	uint64_t kernel_start = start;
	findWheelOffset(kernel_start, wheelidx);

	// set static args
	sclSetKernelArg(pd.getsegprimes, 0, sizeof(uint64_t), &kernel_start);
	sclSetKernelArg(pd.getsegprimes, 1, sizeof(uint64_t), &stop);
	sclSetKernelArg(pd.getsegprimes, 2, sizeof(int32_t), &wheelidx);
	sclSetKernelArg(pd.getsegprimes, 3, sizeof(cl_mem), &d_profileprime);
	sclSetKernelArg(pd.getsegprimes, 4, sizeof(cl_mem), &pd.d_primecount);

	// zero prime count
	sclEnqueueKernel(hardware, pd.clearn);

	// Benchmark the GPU
	double kernel_ms = ProfilesclEnqueueKernel(hardware, pd.getsegprimes);

	// target runtime for prime generator kernel is 1.0 ms
	double prof_multi = 1.0 / kernel_ms;

	// update chunk size based on the profile
	calc_range = (uint64_t)( (double)calc_range * prof_multi );

	// limit kernel global size
	if(calc_range > 4294900000){
		calc_range = 4294900000;
	}

	// get a count of primes in the new gpu worksize
	stop = start + calc_range;

	// check overflow at 2^64
	if(stop < start){
		stop = 0xFFFFFFFFFFFFFFFF;
		calc_range = stop - start;
	}

	range_primes = (stop / log(stop)) - (start / log(start));

	// calculate prime array size based on result
	mem_size = (uint64_t)( 1.5 * (double)range_primes );
	// make it a multiple of check kernel's local size
	mem_size = (mem_size / pd.check.local_size[0]) * pd.check.local_size[0];	

	if(mem_size > UINT32_MAX){
		fprintf(stderr, "ERROR: mem_size too large.\n");
                printf( "ERROR: mem_size too large.\n" );
		exit(EXIT_FAILURE);
	}

	sd.range = calc_range;
	sd.psize = mem_size;

	// free temporary array
	sclReleaseMemObject(d_profileprime);

}


void finalizeResults( workStatus & st ){

	char line[256];
	uint32_t lc = 0;
	FILE * resfile;

	if(st.factorcount){
		// check result file has the same number of lines as the factor count
		resfile = my_fopen(RESULTS_FILENAME,"r");

		if(resfile == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}

		while(fgets(line, sizeof(line), resfile) != NULL) {
			++lc;
		}

		fclose(resfile);

		if(lc < st.factorcount){
			fprintf(stderr,"ERROR: Missing factors in %s !!!\n",RESULTS_FILENAME);
			printf("ERROR: Missing factors in %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}

	// print checksum
	resfile = my_fopen(RESULTS_FILENAME,"a");

	if(resfile == NULL){
		fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
		exit(EXIT_FAILURE);
	}

	if(st.factorcount){
		if( fprintf( resfile, "%016" PRIX64 "\n", st.checksum ) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	else{
		if( fprintf( resfile, "no factors\n%016" PRIX64 "\n", st.checksum ) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}

	fclose(resfile);
}

cl_uint2 getPower(uint32_t prime, uint32_t startN){
	uint32_t totalpower = 0;
	uint64_t currp = prime;
	uint32_t q = startN / currp;
	while(true){
		totalpower += q;
		currp = currp * prime;
		if(currp > startN)break;
		q = startN / currp;
	}
	uint32_t curBit = 0x80000000;
	if(totalpower > 1){
		curBit >>= ( __builtin_clz(totalpower) + 1 );
	}
	return (cl_uint2){totalpower, curBit};
}

// factorial power table
void setupPowerTable(progData & pd, workStatus & st, searchData & sd, sclHard hardware, uint32_t * h_primecount ){

	cl_int err = 0;
	uint32_t stride = 2560000;
	uint32_t start_factorial = st.nmin-1;

	// generate primes for power table
	size_t primelistsize;
	uint32_t *smprime = (uint32_t*)primesieve_generate_primes(2, start_factorial, &primelistsize, UINT32_PRIMES);
	uint64_t tablesize = primelistsize*8;	// cl_ulong or cl_uint2
	cl_uint2 * smpower = (cl_uint2 *)malloc(tablesize);
	if( smpower == NULL ){
		fprintf(stderr,"malloc error: smpower\n");
		exit(EXIT_FAILURE);
	}
	for(uint32_t i=0; i<primelistsize; ++i){
		smpower[i] = getPower(smprime[i], start_factorial);
	}
	cl_ulong * h_prime = (cl_ulong *)malloc(tablesize);
	if( h_prime == NULL ){
		fprintf(stderr,"malloc error: h_prime\n");
		exit(EXIT_FAILURE);
	}
	cl_uint2 * h_power = (cl_uint2 *)malloc(tablesize);
	if( h_power == NULL ){
		fprintf(stderr,"malloc error: h_power\n");
		exit(EXIT_FAILURE);
	}

	// compress the power table by combining primes with the same power
	// skip prime = 2
	h_prime[0] = smprime[0];
	h_power[0] = smpower[0];
	uint32_t m=1;
	for(uint32_t i=1; i<primelistsize; ++m){
		h_prime[m] = smprime[i];
		h_power[m] = smpower[i];
		for(++i; i<primelistsize && h_power[m].s0 == smpower[i].s0; ++i){
			unsigned __int128 pp = (unsigned __int128)h_prime[m] * smprime[i];
			if(pp > 0xFFFFFFFFFFFFFFFF) break;
			h_prime[m] = pp;
		}
	}
	free(smprime);
	free(smpower);
	sd.smcount = m;
	fprintf(stderr,"Compressed %u power table terms to %u\n",(uint32_t)primelistsize,m);
	if(boinc_is_standalone()){
		printf("Compressed %u power table terms to %u\n",(uint32_t)primelistsize,m);
	}

	// send read only prime/power tables to gpu
	tablesize = (uint64_t)m*8;	// cl_ulong or cl_uint2
	if( sd.maxmalloc < tablesize ){
		fprintf(stderr, "ERROR: power table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", tablesize, sd.maxmalloc);
                printf( "ERROR: power table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", tablesize, sd.maxmalloc);
		exit(EXIT_FAILURE);
	}
	pd.d_products = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, tablesize, NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: primeproducts array\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_powers = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, tablesize, NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: Powers array.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWriteNB(hardware, tablesize, pd.d_products, h_prime);
	sclWrite(hardware, tablesize, pd.d_powers, h_power);
	free(h_prime);
	free(h_power);

	// verify the new power table
	sclSetGlobalSize( pd.verifyslow, stride );
	sclSetGlobalSize( pd.verify, stride );
	uint32_t ver_groups = stride / 256;				// 10000
	sclSetGlobalSize( pd.verifyreduce, ver_groups );
	uint32_t red_groups = (ver_groups / 256)+1;			// 40
	sclSetGlobalSize( pd.verifyresult, red_groups );
	cl_mem d_verify = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, ver_groups*sizeof(cl_ulong4), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	sclSetKernelArg(pd.verifyslow, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyslow, 1, sizeof(uint32_t), &start_factorial);

	sclSetKernelArg(pd.verify, 0, sizeof(cl_mem), &pd.d_products);
	sclSetKernelArg(pd.verify, 1, sizeof(cl_mem), &pd.d_powers);
	sclSetKernelArg(pd.verify, 2, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verify, 3, sizeof(uint32_t), &sd.smcount);

	sclSetKernelArg(pd.verifyreduce, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyreduce, 1, sizeof(uint32_t), &ver_groups);

	sclSetKernelArg(pd.verifyresult, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyresult, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.verifyresult, 2, sizeof(uint32_t), &red_groups);

	sclEnqueueKernel(hardware, pd.verifyslow);
	sclEnqueueKernel(hardware, pd.verify);
	sclEnqueueKernel(hardware, pd.verifyreduce);
	sclEnqueueKernel(hardware, pd.verifyresult);

	// copy verification flag to host memory, blocking
	sclRead(hardware, 6*sizeof(uint32_t), pd.d_primecount, h_primecount);
	// flag set if there is a gpu power table error
	if(h_primecount[3] == 1){
		fprintf(stderr,"error: power table verification failed\n");
		printf("error: power table verification failed\n");
		exit(EXIT_FAILURE);
	}
	fprintf(stderr,"Verified power table (%" PRIu64 " bytes) starting sieve...\n", tablesize*2);
	if(boinc_is_standalone()){
		printf("Verified power table (%" PRIu64 " bytes) starting sieve...\n", tablesize*2);
	}
	sclReleaseMemObject(d_verify);
	sclSetKernelArg(pd.setup, 2, sizeof(cl_mem), &pd.d_products);
	sclSetKernelArg(pd.setup, 5, sizeof(cl_mem), &pd.d_powers);
	sclSetKernelArg(pd.setup, 6, sizeof(uint32_t), &start_factorial);

	sd.nlimit = st.nmax;
}

// primorial product and prime tables
void setupPrimeProducts(progData & pd, workStatus & st, searchData & sd, sclHard hardware, uint32_t * h_primecount ){

	cl_int err = 0;
	uint32_t stride = 2560000;
	uint32_t start_primorial = st.nmin-1;
	uint32_t end_primorial = st.nmax-1;
	uint64_t totalprimes = 0;

	size_t smsize;
	uint32_t * smprime = (uint32_t*)primesieve_generate_primes(2, start_primorial, &smsize, UINT32_PRIMES);
	totalprimes+=smsize;

	size_t itersize;
	uint32_t * h_iterprime = (uint32_t*)primesieve_generate_primes(start_primorial+1, end_primorial, &itersize, UINT32_PRIMES);
	totalprimes+=itersize;
	sd.nlimit = itersize;

	uint64_t tablesize = smsize*sizeof(cl_ulong);
	cl_ulong * h_prime = (cl_ulong *)malloc(tablesize);
	if( h_prime == NULL ){
		fprintf(stderr,"malloc error: h_prime\n");
		exit(EXIT_FAILURE);
	}

	// compress the table by combining primes
	uint32_t m=0;
	for(uint32_t i=0; i<smsize; ++m){
		h_prime[m] = smprime[i];
		for(++i; i<smsize; ++i){
			unsigned __int128 pp = (unsigned __int128)h_prime[m] * smprime[i];
			if(pp > 0xFFFFFFFFFFFFFFFF) break;
			h_prime[m] = pp;
		}
	}

	free(smprime);
	sd.smcount = m;
	fprintf(stderr,"Compressed %u primes to %u products\n",(uint32_t)smsize,m);
	if(boinc_is_standalone()){
		printf("Compressed %u primes to %u products\n",(uint32_t)smsize,m);
	}

	// send prime product table to gpu
	tablesize = (uint64_t)m*sizeof(cl_ulong);
	if( sd.maxmalloc < tablesize ){
		fprintf(stderr, "ERROR: prime product table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", tablesize, sd.maxmalloc);
                printf( "ERROR: prime product table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", tablesize, sd.maxmalloc);
		exit(EXIT_FAILURE);
	}
	pd.d_products = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, tablesize, NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: primeproducts array\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWriteNB(hardware, tablesize, pd.d_products, h_prime);

	// send partial prime list to gpu
	uint64_t itertablesize = (uint64_t)itersize*sizeof(cl_uint);
	if( sd.maxmalloc < itertablesize ){
		fprintf(stderr, "ERROR: prime table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", itertablesize, sd.maxmalloc);
                printf( "ERROR: prime table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", itertablesize, sd.maxmalloc);
		exit(EXIT_FAILURE);
	}
	pd.d_smallprimes = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, itertablesize, NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: SmallPrimes array\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWriteNB(hardware, itertablesize, pd.d_smallprimes, h_iterprime);

	// verify product and partial prime tables
	size_t fullprimelistsize;
	uint32_t * fullprimelist = (uint32_t*)primesieve_generate_primes(2, st.nmax, &fullprimelistsize, UINT32_PRIMES);
	if(fullprimelistsize != totalprimes){
		fprintf(stderr, "ERROR: CPU sieve failure.\n");
                printf( "ERROR: CPU sieve failure.\n" );
		exit(EXIT_FAILURE);
	}
	cl_mem d_fullprimelist = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, fullprimelistsize*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWrite(hardware, fullprimelistsize*sizeof(cl_uint), d_fullprimelist, fullprimelist);

	free(h_prime);
	free(h_iterprime);
	free(fullprimelist);

	sclSetGlobalSize( pd.verifyslow, stride );
	sclSetGlobalSize( pd.verify, stride );
	uint32_t ver_groups = stride / 256;				// 10000
	sclSetGlobalSize( pd.verifyreduce, ver_groups );
	uint32_t red_groups = (ver_groups / 256)+1;			// 40
	sclSetGlobalSize( pd.verifyresult, red_groups );
	cl_mem d_verify = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, ver_groups*sizeof(cl_ulong4), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	uint32_t fplsize = (uint32_t)fullprimelistsize;
	sclSetKernelArg(pd.verifyslow, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyslow, 1, sizeof(cl_mem), &d_fullprimelist);
	sclSetKernelArg(pd.verifyslow, 2, sizeof(uint32_t), &fplsize);

	sclSetKernelArg(pd.verify, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verify, 1, sizeof(cl_mem), &pd.d_products);
	sclSetKernelArg(pd.verify, 2, sizeof(cl_mem), &pd.d_smallprimes);
	sclSetKernelArg(pd.verify, 3, sizeof(uint32_t), &sd.smcount);
	sclSetKernelArg(pd.verify, 4, sizeof(uint32_t), &sd.nlimit);

	sclSetKernelArg(pd.verifyreduce, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyreduce, 1, sizeof(uint32_t), &ver_groups);

	sclSetKernelArg(pd.verifyresult, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyresult, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.verifyresult, 2, sizeof(uint32_t), &red_groups);

	sclEnqueueKernel(hardware, pd.verifyslow);
	sclEnqueueKernel(hardware, pd.verify);
	sclEnqueueKernel(hardware, pd.verifyreduce);
	sclEnqueueKernel(hardware, pd.verifyresult);

	// copy verification flag to host memory, blocking
	sclRead(hardware, 6*sizeof(uint32_t), pd.d_primecount, h_primecount);
	// flag set if there is a gpu product/prime table error
	if(h_primecount[3] == 1){
		fprintf(stderr,"error: product/prime table verification failed\n");
		printf("error: product/prime table verification failed\n");
		exit(EXIT_FAILURE);
	}
	fprintf(stderr,"Verified prime (%" PRIu64 " bytes) and product (%" PRIu64 " bytes) tables. starting sieve...\n", itertablesize, tablesize);
	if(boinc_is_standalone()){
		printf("Verified prime (%" PRIu64 " bytes) and product (%" PRIu64 " bytes) tables. starting sieve...\n", itertablesize, tablesize);
	}
	sclReleaseMemObject(d_verify);
	sclReleaseMemObject(d_fullprimelist);

	sclSetKernelArg(pd.setup, 2, sizeof(cl_mem), &pd.d_products);

	sclSetKernelArg(pd.iterate, 5, sizeof(cl_mem), &pd.d_smallprimes);

}

// compositorial product and prime tables
void setupCompositeProducts(progData & pd, workStatus & st, searchData & sd, sclHard hardware, uint32_t * h_primecount, uint32_t * h_iterprime, uint32_t ipsize ){

	cl_int err = 0;
	uint32_t stride = 2560000;
	uint32_t start_compositorial = st.nmin-1;

	size_t smsize;
	uint32_t * smprime = (uint32_t*)primesieve_generate_primes(2, start_compositorial, &smsize, UINT32_PRIMES);

	uint32_t * composites = (uint32_t *)malloc(st.nmin*sizeof(uint32_t));
	if( composites == NULL ){
		fprintf(stderr,"malloc error: composites\n");
		exit(EXIT_FAILURE);
	}

	uint32_t csize=0;
	for(uint32_t i=0,n=2; n<st.nmin; ++n){
		if(n == smprime[i]){
			++i;
			continue;
		}
		composites[csize++] = n;
	}

	free(smprime);

	uint64_t tablesize = csize*sizeof(cl_ulong);
	cl_ulong * h_comp = (cl_ulong *)malloc(tablesize);
	if( h_comp == NULL ){
		fprintf(stderr,"malloc error: h_comp\n");
		exit(EXIT_FAILURE);
	}

	// compress the table by combining composites
	uint32_t m=0;
	for(uint32_t i=0; i<csize; ++m){
		h_comp[m] = composites[i];
		for(++i; i<csize; ++i){
			unsigned __int128 cc = (unsigned __int128)h_comp[m] * composites[i];
			if(cc > 0xFFFFFFFFFFFFFFFF) break;
			h_comp[m] = cc;
		}
	}

	free(composites);

	sd.smcount = m;
	fprintf(stderr,"Compressed %u composites to %u products\n",(uint32_t)csize,m);
	if(boinc_is_standalone()){
		printf("Compressed %u composites to %u products\n",(uint32_t)csize,m);
	}

	// send read only composite product table to gpu
	tablesize = (uint64_t)m*sizeof(cl_ulong);
	if( sd.maxmalloc < tablesize ){
		fprintf(stderr, "ERROR: composite product table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", tablesize, sd.maxmalloc);
                printf( "ERROR: composite product table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", tablesize, sd.maxmalloc);
		exit(EXIT_FAILURE);
	}
	pd.d_products = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, tablesize, NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: primeproducts array\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWriteNB(hardware, tablesize, pd.d_products, h_comp);

	// send partial prime list to gpu
	uint64_t itertablesize = (uint64_t)ipsize*sizeof(cl_uint);
	if( sd.maxmalloc < itertablesize ){
		fprintf(stderr, "ERROR: prime table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", itertablesize, sd.maxmalloc);
	        printf( "ERROR: prime table size is %" PRIu64 " bytes.  Device supports allocation up to %" PRIu64 " bytes.\n", itertablesize, sd.maxmalloc);
		exit(EXIT_FAILURE);
	}
	pd.d_smallprimes = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, itertablesize, NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: SmallPrimes array\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWriteNB(hardware, itertablesize, pd.d_smallprimes, h_iterprime);

	// verify product and partial prime tables
	size_t fullprimelistsize;
	uint32_t * fullprimelist = (uint32_t*)primesieve_generate_primes(2, st.nmax, &fullprimelistsize, UINT32_PRIMES);
	cl_mem d_fullprimelist = clCreateBuffer( hardware.context, CL_MEM_READ_ONLY, fullprimelistsize*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	sclWrite(hardware, fullprimelistsize*sizeof(cl_uint), d_fullprimelist, fullprimelist);
	free(h_comp);
	free(fullprimelist);

	sclSetGlobalSize( pd.verifyslow, stride );
	sclSetGlobalSize( pd.verify, stride );
	uint32_t ver_groups = stride / 256;				// 10000
	sclSetGlobalSize( pd.verifyreduce, ver_groups );
	uint32_t red_groups = (ver_groups / 256)+1;			// 40
	sclSetGlobalSize( pd.verifyresult, red_groups );
	cl_mem d_verify = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, ver_groups*sizeof(cl_ulong4), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	uint32_t fplsize = (uint32_t)fullprimelistsize;
	sclSetKernelArg(pd.verifyslow, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyslow, 1, sizeof(cl_mem), &d_fullprimelist);
	sclSetKernelArg(pd.verifyslow, 2, sizeof(uint32_t), &fplsize);
	sclSetKernelArg(pd.verifyslow, 3, sizeof(uint32_t), &st.nmax);

	sclSetKernelArg(pd.verify, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verify, 1, sizeof(cl_mem), &pd.d_products);
	sclSetKernelArg(pd.verify, 2, sizeof(cl_mem), &pd.d_smallprimes);
	sclSetKernelArg(pd.verify, 3, sizeof(uint32_t), &sd.smcount);
	sclSetKernelArg(pd.verify, 4, sizeof(uint32_t), &ipsize);
	sclSetKernelArg(pd.verify, 5, sizeof(uint32_t), &st.nmin);
	sclSetKernelArg(pd.verify, 6, sizeof(uint32_t), &st.nmax);

	sclSetKernelArg(pd.verifyreduce, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyreduce, 1, sizeof(uint32_t), &ver_groups);

	sclSetKernelArg(pd.verifyresult, 0, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyresult, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.verifyresult, 2, sizeof(uint32_t), &red_groups);

	sclEnqueueKernel(hardware, pd.verifyslow);
	sclEnqueueKernel(hardware, pd.verify);
	sclEnqueueKernel(hardware, pd.verifyreduce);
	sclEnqueueKernel(hardware, pd.verifyresult);

	// copy verification flag to host memory, blocking
	sclRead(hardware, 6*sizeof(uint32_t), pd.d_primecount, h_primecount);
	// flag set if there is a gpu product/prime table error
	if(h_primecount[3] == 1){
		fprintf(stderr,"error: product/prime table verification failed\n");
		printf("error: product/prime table verification failed\n");
		exit(EXIT_FAILURE);
	}
	fprintf(stderr,"Verified prime (%" PRIu64 " bytes) and product (%" PRIu64 " bytes) tables. starting sieve...\n", itertablesize, tablesize);
	if(boinc_is_standalone()){
		printf("Verified prime (%" PRIu64 " bytes) and product (%" PRIu64 " bytes) tables. starting sieve...\n", itertablesize, tablesize);
	}
	sclReleaseMemObject(d_verify);
	sclReleaseMemObject(d_fullprimelist);

	sclSetKernelArg(pd.setup, 2, sizeof(cl_mem), &pd.d_products);
	sclSetKernelArg(pd.setup, 5, sizeof(uint32_t), &start_compositorial);

	sclSetKernelArg(pd.iterate, 5, sizeof(cl_mem), &pd.d_smallprimes);

	sd.nlimit = st.nmax;

}

void cl_sieve( sclHard hardware, workStatus & st, searchData & sd ){

	progData pd = {};
	bool first_iteration = true;
	time_t boinc_last, ckpt_last, time_curr;
	cl_int err = 0;

	// setup kernel parameters
	setupSearch(st,sd);

	// device arrays
	pd.d_primecount = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, 6*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
        pd.d_factor = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numresults*sizeof(factor), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: d_factor array.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	// build kernels
        pd.clearn = sclGetCLSoftware(clearn_cl,"clearn",hardware, NULL);
        pd.clearresult = sclGetCLSoftware(clearresult_cl,"clearresult",hardware, NULL);
        pd.addsmallprimes = sclGetCLSoftware(addsmallprimes_cl,"addsmallprimes",hardware, NULL);
	if(st.pmax < 0xFFFFFFFFFF000000){
	        pd.getsegprimes = sclGetCLSoftware(getsegprimes_cl,"getsegprimes",hardware, NULL);
	}
	else{
	       	pd.getsegprimes = sclGetCLSoftware(getsegprimes_cl,"getsegprimes",hardware, "-D CKOVERFLOW=1" );
	}
	if(st.factorial){
		pd.setup = sclGetCLSoftware(setup_cl,"setup",hardware, NULL);
		pd.iterate = sclGetCLSoftware(iterate_cl,"iterate",hardware, NULL);
		pd.check = sclGetCLSoftware(check_cl,"check",hardware, NULL);
		pd.verifyslow = sclGetCLSoftware(verifyslow_cl,"verifyslow",hardware, NULL);
		pd.verify = sclGetCLSoftware(verify_cl,"verify",hardware, NULL);

	}
	else if(st.primorial){
		pd.setup = sclGetCLSoftware(primsetup_cl,"primsetup",hardware, NULL);
		pd.iterate = sclGetCLSoftware(primiterate_cl,"primiterate",hardware, NULL);
		pd.check = sclGetCLSoftware(primcheck_cl,"primcheck",hardware, NULL);
		pd.verifyslow = sclGetCLSoftware(primverifyslow_cl,"primverifyslow",hardware, NULL);
		pd.verify = sclGetCLSoftware(primverify_cl,"primverify",hardware, NULL);
	}
	else if(st.compositorial){
		pd.setup = sclGetCLSoftware(compsetup_cl,"compsetup",hardware, NULL);
		pd.iterate = sclGetCLSoftware(compiterate_cl,"compiterate",hardware, NULL);
		pd.check = sclGetCLSoftware(check_cl,"check",hardware, NULL);
		pd.verifyslow = sclGetCLSoftware(compverifyslow_cl,"compverifyslow",hardware, NULL);
		pd.verify = sclGetCLSoftware(compverify_cl,"compverify",hardware, NULL);
	}
	pd.verifyreduce = sclGetCLSoftware(verifyreduce_cl,"verifyreduce",hardware, NULL);
	pd.verifyresult = sclGetCLSoftware(verifyresult_cl,"verifyresult",hardware, NULL);

	if(pd.verifyslow.local_size[0] != 256){
		pd.verifyslow.local_size[0] = 256;
		fprintf(stderr, "Set verifyslow kernel local size to 256\n");
	}
	if(pd.verify.local_size[0] != 256){
		pd.verify.local_size[0] = 256;
		fprintf(stderr, "Set verifypow kernel local size to 256\n");
	}
	if(pd.verifyreduce.local_size[0] != 256){
		pd.verifyreduce.local_size[0] = 256;
		fprintf(stderr, "Set verifyreduce kernel local size to 256\n");
	}
	if(pd.verifyresult.local_size[0] != 256){
		pd.verifyresult.local_size[0] = 256;
		fprintf(stderr, "Set verifyresult kernel local size to 256\n");
	}

	// kernels have __attribute__ ((reqd_work_group_size(256, 1, 1)))
	// it's still possible the CL complier picked a different size
	if(pd.getsegprimes.local_size[0] != 256){
		pd.getsegprimes.local_size[0] = 256;
		fprintf(stderr, "Set getsegprimes kernel local size to 256\n");
	}
	if(pd.check.local_size[0] != 256){
		pd.check.local_size[0] = 256;
		fprintf(stderr, "Set check kernel local size to 256\n");
	}


	if( sd.test ){
		// clear result file
		FILE * temp_file = my_fopen(RESULTS_FILENAME,"w");
		if (temp_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
		fclose(temp_file);
	}
	else{
		// Resume from checkpoint if there is one
		if( read_state( st, sd ) ){
			if(boinc_is_standalone()){
				printf("Current p: %" PRIu64 "\n", st.p);
			}
			fprintf(stderr,"Resuming from checkpoint, current p: %" PRIu64 "\n", st.p);

			//trying to resume a finished workunit
			if( st.p == st.pmax ){
				if(boinc_is_standalone()){
					printf("Workunit complete.\n");
				}
				fprintf(stderr,"Workunit complete.\n");
				boinc_finish(EXIT_SUCCESS);
			}
		}
		// starting from beginning
		else{
			// clear result file
			FILE * temp_file = my_fopen(RESULTS_FILENAME,"w");
			if (temp_file == NULL){
				fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
				exit(EXIT_FAILURE);
			}
			fclose(temp_file);

			// setup boinc trickle up
			st.last_trickle = (uint64_t)time(NULL);
		}
	}

	// kernel used in profileGPU, setup arg
	sclSetKernelArg(pd.clearn, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetGlobalSize( pd.clearn, 64 );

	profileGPU(pd,st,sd,hardware);

	// number of gpu workgroups, used to size the sum array on gpu
	sd.numgroups = (sd.psize / pd.check.local_size[0]) + 1;

	// host arrays used for data transfer from gpu during checkpoints
	uint64_t * h_checksum = (uint64_t *)malloc(sd.numgroups*sizeof(uint64_t));
	if( h_checksum == NULL ){
		fprintf(stderr,"malloc error: h_checksum\n");
		exit(EXIT_FAILURE);
	}
	uint32_t * h_primecount = (uint32_t *)malloc(6*sizeof(uint32_t));
	if( h_primecount == NULL ){
		fprintf(stderr,"malloc error: h_primecount\n");
		exit(EXIT_FAILURE);
	}

	// array of primes or composites used during CPU factor verification
	uint32_t *verifylist = NULL;
	size_t verifylistsize = 0;
	if(st.primorial){
		verifylist = (uint32_t*)primesieve_generate_primes(103, st.nmax, &verifylistsize, UINT32_PRIMES);
	}
	else if(st.compositorial){
		size_t allprimesize;
		uint32_t * allprime = (uint32_t*)primesieve_generate_primes(45, st.nmax, &allprimesize, UINT32_PRIMES);
		verifylist = (uint32_t *)malloc(st.nmax*sizeof(uint32_t));
		if( verifylist == NULL ){
			fprintf(stderr,"malloc error: verifylist\n");
			exit(EXIT_FAILURE);
		}
		uint32_t csize=0;
		for(uint32_t i=0,n=45; n<st.nmax; ++n){
			if(n == allprime[i]){
				++i;
				continue;
			}
			verifylist[csize++] = n;
		}
		free(allprime);
		verifylistsize = csize;
	}

	// array of primes from nmin to nmax+prime gap
	uint32_t * h_iterprime = NULL;
	size_t itersize = 0;
	if(st.compositorial){ 
		h_iterprime = (uint32_t*)primesieve_generate_primes(st.nmin, st.nmax+320, &itersize, UINT32_PRIMES);
	}

	sclSetGlobalSize( pd.getsegprimes, (sd.range/60)+1 );
	sclSetGlobalSize( pd.addsmallprimes, 64 );
	sclSetGlobalSize( pd.setup, sd.psize );
	sclSetGlobalSize( pd.iterate, sd.psize );
	sclSetGlobalSize( pd.check, sd.psize );
	sclSetGlobalSize( pd.clearresult, sd.numgroups );

	pd.d_primes = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, sd.psize*sizeof(cl_ulong8), NULL, &err);
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
        pd.d_sum = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numgroups*sizeof(cl_ulong), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	// set static kernel args
	sclSetKernelArg(pd.clearresult, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.clearresult, 1, sizeof(cl_mem), &pd.d_sum);
	sclSetKernelArg(pd.clearresult, 2, sizeof(uint32_t), &sd.numgroups);

	sclSetKernelArg(pd.getsegprimes, 3, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.getsegprimes, 4, sizeof(cl_mem), &pd.d_primecount);

	sclSetKernelArg(pd.addsmallprimes, 2, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.addsmallprimes, 3, sizeof(cl_mem), &pd.d_primecount);

	sclSetKernelArg(pd.setup, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.setup, 1, sizeof(cl_mem), &pd.d_primecount);

	sclSetKernelArg(pd.iterate, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.iterate, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.iterate, 2, sizeof(cl_mem), &pd.d_factor);

	sclSetKernelArg(pd.check, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.check, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.check, 2, sizeof(cl_mem), &pd.d_sum);
	if(st.factorial || st.compositorial){
		uint32_t lastn = st.nmax-1;
		sclSetKernelArg(pd.check, 3, sizeof(uint32_t), &lastn);
	}

	time(&boinc_last);
	time(&ckpt_last);
	time_t totals, totalf;
	if(boinc_is_standalone()){
		time(&totals);
	}

	float kernel_ms;
	int kernelq = 0;
	const int maxq = sd.compute ? 20 : 100;		// target kernel queue depth is 1 second
	cl_event launchEvent = NULL;
	const double irsize = 1.0 / (double)(st.pmax-st.pmin);

	sclEnqueueKernel(hardware, pd.clearresult);

	// main search loop
	while(st.p < st.pmax){

		uint64_t stop = st.p + sd.range;
		if(stop > st.pmax || stop < st.p){
			// ck overflow
			stop = st.pmax;
		}

		// clear prime count
		sclEnqueueKernel(hardware, pd.clearn);
		
		time(&time_curr);
		if( ((int)time_curr - (int)boinc_last) > 1 ){
			// update BOINC fraction done every 2 sec
    			double fd = (double)(st.p-st.pmin)*irsize;
			boinc_fraction_done(fd);
			if(boinc_is_standalone()) printf("Sieve Progress: %.1f%%\n",fd*100.0);
			boinc_last = time_curr;
			if( ((int)time_curr - (int)ckpt_last) > 60 ){
				// 1 minute checkpoint
				if(kernelq > 0){
					waitOnEvent(hardware, launchEvent);
					kernelq = 0;
				}
				sleepCPU(hardware);
				boinc_begin_critical_section();
				getResults(pd, st, sd, hardware, h_checksum, h_primecount, verifylist, verifylistsize);
				checkpoint(st, sd);
				boinc_end_critical_section();
				ckpt_last = time_curr;
				// clear result arrays
				sclEnqueueKernel(hardware, pd.clearresult);
			}
		}

		// add small primes that cannot be generated with getsegprimes kernel
		if(st.p < 114){
			uint64_t stop_sm = (stop > 114) ? 114 : stop;
			sclSetKernelArg(pd.addsmallprimes, 0, sizeof(uint64_t), &st.p);
			sclSetKernelArg(pd.addsmallprimes, 1, sizeof(uint64_t), &stop_sm);
			sclEnqueueKernel(hardware, pd.addsmallprimes);
			st.p = stop_sm;
		}

		// get a segment of primes (2-PRPs).  very fast, target kernel time is 1ms
		int32_t wheelidx;
		uint64_t kernel_start = st.p;
		findWheelOffset(kernel_start, wheelidx);

		sclSetKernelArg(pd.getsegprimes, 0, sizeof(uint64_t), &kernel_start);
		sclSetKernelArg(pd.getsegprimes, 1, sizeof(uint64_t), &stop);
		sclSetKernelArg(pd.getsegprimes, 2, sizeof(int32_t), &wheelidx);
		sclEnqueueKernel(hardware, pd.getsegprimes);

		uint32_t sstart = 0;
		uint32_t smax;
		uint32_t nstart = (st.factorial || st. compositorial) ? st.nmin : 0;
		uint32_t nmax;
		uint32_t nextprimepos = 0;

		// setup power table, then profile setup kernel once at program start.  adjust work size to target kernel runtime.
		if(first_iteration){
			if(st.factorial){
				setupPowerTable(pd, st, sd, hardware, h_primecount);
			}
			else if(st.primorial){
				setupPrimeProducts(pd, st, sd, hardware, h_primecount);
			}
			else if(st.compositorial){
				setupCompositeProducts(pd, st, sd, hardware, h_primecount, h_iterprime, itersize);
			}
			smax = sstart + sd.sstep;
			if(smax > sd.smcount)smax = sd.smcount;
			sclSetKernelArg(pd.setup, 3, sizeof(uint32_t), &sstart);
			sclSetKernelArg(pd.setup, 4, sizeof(uint32_t), &smax);
			kernel_ms = ProfilesclEnqueueKernel(hardware, pd.setup);
			sstart += sd.sstep;
			double multi = sd.compute ? 50.0/kernel_ms : 20.0/kernel_ms;	// target kernel time 50ms or 20ms, first iterations have large powers, avg kernel time is less
			uint32_t new_sstep = (uint32_t)( multi * (double)sd.sstep );
			if(!new_sstep) new_sstep=1;
			sd.sstep = new_sstep;
		}

		// setup residue for nmin# / nmin! mod P
		for(; sstart < sd.smcount; sstart += sd.sstep){
			smax = sstart + sd.sstep;
			if(smax > sd.smcount)smax = sd.smcount;
			sclSetKernelArg(pd.setup, 3, sizeof(uint32_t), &sstart);
			sclSetKernelArg(pd.setup, 4, sizeof(uint32_t), &smax);
			if(kernelq == 0){
				launchEvent = sclEnqueueKernelEvent(hardware, pd.setup);
			}
			else{
				sclEnqueueKernel(hardware, pd.setup);
			}
			if(++kernelq == maxq){
				// limit cl queue depth and sleep cpu
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
		}

		// profile iterate kernel once at program start.  adjust work size to target kernel runtime.
		if(first_iteration){
			first_iteration = false;
			if(st.compositorial){
				sclSetKernelArg(pd.iterate, 6, sizeof(uint32_t), &nextprimepos);
			}
			nmax = nstart + sd.nstep;
			if(nmax > sd.nlimit)nmax = sd.nlimit;
			sclSetKernelArg(pd.iterate, 3, sizeof(uint32_t), &nstart);
			sclSetKernelArg(pd.iterate, 4, sizeof(uint32_t), &nmax);
			kernel_ms = ProfilesclEnqueueKernel(hardware, pd.iterate);
			nstart += sd.nstep;
			double multi = sd.compute ? 50.0/kernel_ms : 10.0/kernel_ms;	// target kernel time 50ms or 10ms
			uint32_t new_nstep = (uint32_t)( multi * (double)sd.nstep );
			if(!new_nstep) new_nstep=1;
			sd.nstep = new_nstep;
			fprintf(stderr,"c:%u u:%u t:%u r:%u p:%u s:%u n:%u\n", (uint32_t)sd.compute, sd.computeunits, sd.threadcount, sd.range, sd.psize, sd.sstep, sd.nstep);
			if(boinc_is_standalone()){
				printf("c:%u u:%u t:%u r:%u p:%u s:%u n:%u\n", (uint32_t)sd.compute, sd.computeunits, sd.threadcount, sd.range, sd.psize, sd.sstep, sd.nstep);
			}
		}

		// iterate from nmin# / nmin! to nmax# / nmax-1! mod P
		for(; nstart < sd.nlimit; nstart += sd.nstep){
			if(st.compositorial){
				while(h_iterprime[nextprimepos] < nstart){
					++nextprimepos;
				}
				sclSetKernelArg(pd.iterate, 6, sizeof(uint32_t), &nextprimepos);
			}
			nmax = nstart + sd.nstep;
			if(nmax > sd.nlimit)nmax = sd.nlimit;
			sclSetKernelArg(pd.iterate, 3, sizeof(uint32_t), &nstart);
			sclSetKernelArg(pd.iterate, 4, sizeof(uint32_t), &nmax);
			if(kernelq == 0){
				launchEvent = sclEnqueueKernelEvent(hardware, pd.iterate);
			}
			else{
				sclEnqueueKernel(hardware, pd.iterate);
			}
			if(++kernelq == maxq){
				// limit cl queue depth and sleep cpu
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
		}

		// checksum kernel
		sclEnqueueKernel(hardware, pd.check);

		uint64_t nextp = st.p + sd.range;
		if(nextp < st.p){
			// ck overflow at 2^64
			break;
		}
		else{
			st.p = nextp;
		}

	}

	// final checkpoint
	if(kernelq > 0){
		waitOnEvent(hardware, launchEvent);
	}
	sleepCPU(hardware);

	boinc_begin_critical_section();
	st.p = st.pmax;
	boinc_fraction_done(1.0);
	if(boinc_is_standalone()) printf("Sieve Progress: %.1f%%\n",100.0);
	getResults(pd, st, sd, hardware, h_checksum, h_primecount, verifylist, verifylistsize);
	checkpoint(st, sd);
	finalizeResults(st);
	boinc_end_critical_section();

	fprintf(stderr,"Sieve complete.\nfactors %" PRIu64 ", prime count %" PRIu64 "\n", st.factorcount, st.primecount);

	if(boinc_is_standalone()){
		time(&totalf);
		printf("Sieve finished in %d sec.\n", (int)totalf - (int)totals);
		printf("factors %" PRIu64 ", prime count %" PRIu64 ", checksum %016" PRIX64 "\n", st.factorcount, st.primecount, st.checksum);
	}

	free(h_checksum);
	free(h_primecount);
	cleanup(pd, sd, st);
	if(st.primorial){
		free(verifylist);
	}
	else if(st.compositorial){
		free(h_iterprime);
		free(verifylist);
	}
}


void run_test( sclHard hardware, workStatus & st, searchData & sd ){

	int goodtest = 0;

	printf("Beginning self test of 12 ranges.\n");

	time_t start, finish;
	time(&start);

	printf("Starting Factorial tests\n\n");
//	-p 100e6 -P 101e6 -n 1e6 -N 2e6 -!
	st.factorial = true;
	st.primorial = false;
	st.compositorial = false;
	st.pmin = 100000000;
	st.pmax = 101000000;
	st.nmin = 1000000;
	st.nmax = 2000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 1071 && st.primecount == 54211 && st.checksum == 0x000004F844B5103C ){
		printf("test case 1 passed.\n\n");
		fprintf(stderr,"test case 1 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 1 failed.\n\n");
		fprintf(stderr,"test case 1 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 1e12 -P 100001e7 -n 10000 -N 2e6 -!
	st.factorial = true;
	st.primorial = false;
	st.compositorial = false;
	st.pmin = 1000000000000;
	st.pmax = 1000010000000;
	st.nmin = 10000;
	st.nmax = 2000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 3 && st.primecount == 361727 && st.checksum == 0x0505A1C238896511 ){
		printf("test case 2 passed.\n\n");
		fprintf(stderr,"test case 2 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 2 failed.\n\n");
		fprintf(stderr,"test case 2 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 101 -P 100000 -n 101 -N 1e6 -!
	st.factorial = true;
	st.primorial = false;
	st.compositorial = false;
	st.pmin = 101;
	st.pmax = 100000;
	st.nmin = 101;
	st.nmax = 1000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 42821 && st.primecount == 9571 && st.checksum == 0x0000000065DDB8A0 ){
		printf("test case 3 passed.\n\n");
		fprintf(stderr,"test case 3 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 3 failed.\n\n");
		fprintf(stderr,"test case 3 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 1e12 -P 1000001e6 -n 10e7 -N 11e7 -!
	st.factorial = true;
	st.primorial = false;
	st.compositorial = false;
	st.pmin = 1000000000000;
	st.pmax = 1000001000000;
	st.nmin = 100000000;
	st.nmax = 110000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 3 && st.primecount == 36249 && st.checksum == 0x00804FE7D7AA6C09 ){
		printf("test case 4 passed.\n\n");
		fprintf(stderr,"test case 4 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 4 failed.\n\n");
		fprintf(stderr,"test case 4 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

	printf("Starting Primorial tests\n\n");
//	-p 100e6 -P 101e6 -n 101 -N 25e6 -#
	st.factorial = false;
	st.primorial = true;
	st.compositorial = false;
	st.pmin = 100000000;
	st.pmax = 101000000;
	st.nmin = 101;
	st.nmax = 25000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 1703 && st.primecount == 54211 && st.checksum == 0x0000027EFF497990 ){
		printf("test case 5 passed.\n\n");
		fprintf(stderr,"test case 5 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 5 failed.\n\n");
		fprintf(stderr,"test case 5 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 101 -P 2e6 -n 101 -N 2e6 -#
	st.factorial = false;
	st.primorial = true;
	st.compositorial = false;
	st.pmin = 101;
	st.pmax = 2000000;
	st.nmin = 101;
	st.nmax = 2000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 24503 && st.primecount == 148954 && st.checksum == 0x000000027BF5B8E0 ){
		printf("test case 6 passed.\n\n");
		fprintf(stderr,"test case 6 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 6 failed.\n\n");
		fprintf(stderr,"test case 6 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 1e11 -P 100005e6 -n 9e6 -N 11e7 -#
	st.factorial = false;
	st.primorial = true;
	st.compositorial = false;
	st.pmin = 100000000000;
	st.pmax = 100005000000;
	st.nmin = 9000000;
	st.nmax = 110000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 32 && st.primecount == 197222 && st.checksum == 0x0022FE7C09210B4B ){
		printf("test case 7 passed.\n\n");
		fprintf(stderr,"test case 7 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 7 failed.\n\n");
		fprintf(stderr,"test case 7 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-n 600000 -N 30e6 -p 1730720716e6 -P 1730720720e6 -#
	st.factorial = false;
	st.primorial = true;
	st.compositorial = false;
	st.pmin = 1730720716000000;
	st.pmax = 1730720720000000;
	st.nmin = 600000;
	st.nmax = 30000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 1 && st.primecount == 114208 && st.checksum == 0x5CDCB47F7E9532C2 ){
		printf("test case 8 passed.\n\n");
		fprintf(stderr,"test case 8 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 8 failed.\n\n");
		fprintf(stderr,"test case 8 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

	printf("Starting Compositorial tests\n\n");
//	-p 200e6 -P 20001e4 -n 101 -N 26e6 -c
	st.factorial = false;
	st.primorial = false;
	st.compositorial = true;
	st.pmin = 200000000;
	st.pmax = 200010000;
	st.nmin = 101;
	st.nmax = 26000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 127 && st.primecount == 529 && st.checksum == 0x0000001848D8AFBB ){
		printf("test case 9 passed.\n\n");
		fprintf(stderr,"test case 9 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 9 failed.\n\n");
		fprintf(stderr,"test case 9 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 101 -P 1e5 -n 101 -N 1e6 -c
	st.factorial = false;
	st.primorial = false;
	st.compositorial = true;
	st.pmin = 101;
	st.pmax = 100000;
	st.nmin = 101;
	st.nmax = 1000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 34271 && st.primecount == 9571 && st.checksum == 0x000000006FF88EAE ){
		printf("test case 10 passed.\n\n");
		fprintf(stderr,"test case 10 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 10 failed.\n\n");
		fprintf(stderr,"test case 10 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-p 2e11 -P 200005e6 -n 15e6 -N 2e7 -c
	st.factorial = false;
	st.primorial = false;
	st.compositorial = true;
	st.pmin = 200000000000;
	st.pmax = 200005000000;
	st.nmin = 15000000;
	st.nmax = 20000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 13 && st.primecount == 192386 && st.checksum == 0x0088B59C23CD3E2B ){
		printf("test case 11 passed.\n\n");
		fprintf(stderr,"test case 11 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 11 failed.\n\n");
		fprintf(stderr,"test case 11 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	-n 700000 -N 25e6 -p 1e12 -P 1000001e6 -c
	st.factorial = false;
	st.primorial = false;
	st.compositorial = true;
	st.pmin = 1000000000000;
	st.pmax = 1000001000000;
	st.nmin = 700000;
	st.nmax = 25000000;
	cl_sieve( hardware, st, sd );
	if( st.factorcount == 2 && st.primecount == 36249 && st.checksum == 0x0080997AF3BF42FE ){
		printf("test case 12 passed.\n\n");
		fprintf(stderr,"test case 12 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 12 failed.\n\n");
		fprintf(stderr,"test case 12 failed.\n");
	}
	st.checksum = 0;
	st.primecount = 0;
	st.factorcount = 0;

//	done
	if(goodtest == 12){
		printf("All test cases completed successfully!\n");
		fprintf(stderr, "All test cases completed successfully!\n");
	}
	else{
		printf("Self test FAILED!\n");
		fprintf(stderr, "Self test FAILED!\n");
	}

	time(&finish);
	printf("Elapsed time: %d sec.\n", (int)finish - (int)start);

}


