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
#include <getopt.h>
#include <omp.h>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"
#include "primesieve.h"
#include "putil.h"
#include "cl_sieve.h"

void help()
{
	printf("Welcome to PFCSieve, an OpenCL program to find factors of primorial, factorial, and compositorial prime candidates of the form n#+-1, n!+-1, and n!/#+-1\n");
	printf("Program usage:\n");
	printf("-!	Use factorial mode\n");
	printf("-#	Use primorial mode\n");
	printf("-c	Use compositorial mode\n");
	printf("-n #	Start primorial n#+-1, factorial n!+-1, or compositorial n!/#+-1\n");
	printf("-N #	End primorial N#+-1, factorial N!+-1, or compositorial N!/#+-1\n");
	printf("		N range is 101 <= -n < -N < 2^31, [-n, -N) exclusive\n");
	printf("-p #	Starting prime factor p\n");
	printf("-P #	End prime factor P\n");
	printf("		P range is 3 <= -p < -P < 2^64, [-p, -P) exclusive\n");
	printf("		Note for primorial and factorial -p must be >= -n because there are no factors when p <= n\n");
	printf("		Note N!+-1, N#+-1, and N!/#+-1 are not divisible by 2.\n");
	printf("-v #	Optional, specify the number of CPU threads used to verify factors.  Default is 2, max is 128.\n");
	printf("-s 	Perform self test to verify proper operation of the program with the current GPU.\n");
	printf("-h	Print this help\n");
        boinc_finish(EXIT_FAILURE);
}


static const char *short_opts = "p:P:n:N:v:s!#cfh";

static int parse_option(int opt, char *arg, const char *source, workStatus & st, searchData & sd)
{
  int status = 0;

  switch (opt)
  {
    case 'p':
      status = parse_uint64(&st.pmin,arg,3,0xFFFFFFFFFFFFFFFF-1);
      break;

    case 'P':
      status = parse_uint64(&st.pmax,arg,4,0xFFFFFFFFFFFFFFFF);
      break;
      
    case 'n':
      status = parse_uint(&st.nmin,arg,101,0x7FFFFFFF-1);
      break;

    case 'N':
      status = parse_uint(&st.nmax,arg,102,0x7FFFFFFF);
      break;

    case 'v':
      status = parse_uint(&sd.threadcount,arg,1,128);
      break;

    case 's':
      sd.test = true;
      fprintf(stderr,"Performing self test.\n");
      printf("Performing self test.\n");
      break;

    case '!':
      st.factorial = true;
      fprintf(stderr,"-! argument specified for factorial mode.\n");
      printf("\n-! argument specified for factorial mode.\n\n");
      break;

    case '#':
      st.primorial = true;
      fprintf(stderr,"-# argument specified for primorial mode.\n");
      printf("\n-# argument specified for primorial mode.\n\n");
      break;

    case 'c':
      st.compositorial = true;
      fprintf(stderr,"-c argument specified for compositorial mode.\n");
      printf("\n-c argument specified for compositorial mode.\n\n");
      break;

    case 'h':
      help();
      break;

    case '?':
      help();
      break;
  }

  return status;
}

static const struct option long_opts[] = {
  {"device",  optional_argument, 0, 'd'},		// handle --device arg, but it's not used
  {"test",  no_argument, 0, 's'},
  {0,0,0,0}
};


/* Process command-line options using getopt_long().
   Non-option arguments are treated as if they belong to option zero.
   Returns the number of options processed.
 */
static int process_args(int argc, char *argv[], workStatus & st, searchData & sd)
{
  int count = 0, ind = -1, opt;

  while ((opt = getopt_long(argc,argv,short_opts,long_opts,&ind)) != -1)
    switch (parse_option(opt,optarg,NULL,st,sd))
    {
      case 0:
        ind = -1;
        count++;
        break;

      case -1:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: invalid argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: invalid argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
     	  printf("%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      case -2:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: out of range argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: out of range argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
          printf("%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      default:
        printf("unknown command line argument\n");
        boinc_finish(EXIT_FAILURE);
    }

  while (optind < argc)
    switch (parse_option(0,argv[optind],NULL,st,sd))
    {
      case 0:
        optind++;
        count++;
        break;

      case -1:
        fprintf(stderr,"%s: invalid non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      case -2:
        fprintf(stderr,"%s: out of range non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      default:
        boinc_finish(EXIT_FAILURE);
    }


  return count;
}


#ifdef _WIN32
double getSysOpType()
{
    double ret = 0.0;
    NTSTATUS(WINAPI *RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        ret = (double)osInfo.dwMajorVersion;
    }
    return ret;
}
#endif


int main(int argc, char *argv[])
{ 
	sclHard hardware = {};
	searchData sd = {};
	sd.numresults = 1000000;
	sd.write_state_a_next = true;
	sd.threadcount = 2;
	workStatus st = {};

        // Initialize BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
        options.normal_thread_priority = true;
        boinc_init_options(&options);

	fprintf(stderr, "\nPFCSieve v%s.%s by Bryan Little\nwith contributions by Yves Gallot, Mark Rodenkirch, and Kim Walisch\n",VERSION_MAJOR,VERSION_MINOR);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	if(boinc_is_standalone()){
		printf("\nPFCSieve v%s.%s by Bryan Little\nwith contributions by Yves Gallot, Mark Rodenkirch, and Kim Walisch\n",VERSION_MAJOR,VERSION_MINOR);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	}

        // Print out cmd line for diagnostics
        fprintf(stderr, "Command line: ");
        for (int i = 0; i < argc; i++)
        	fprintf(stderr, "%s ", argv[i]);
        fprintf(stderr, "\n");

	process_args(argc,argv,st,sd);

	omp_set_num_threads(sd.threadcount);

	primesieve_set_num_threads(1);

	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context ctx;
	cl_command_queue queue;
	cl_int err = 0;

	int retval = 0;
	retval = boinc_get_opencl_ids(argc, argv, 0, &device, &platform);
	if (retval) {
		if(boinc_is_standalone()){
			printf("init_data.xml not found, using device 0.\n");

			err = clGetPlatformIDs(1, &platform, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetPlatformIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetPlatformIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetDeviceIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetDeviceIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
		}
		else{
			fprintf(stderr, "Error: boinc_get_opencl_ids() failed with error %d\n", retval );
			exit(EXIT_FAILURE);
		}
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: clCreateContext() returned %d\n", err);
        	exit(EXIT_FAILURE); 
   	}

	// OpenCL v2.0
	//cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	//queue = clCreateCommandQueueWithProperties(ctx, device, qp, &err);

	queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);	
	if(err != CL_SUCCESS) { 
		fprintf(stderr, "Error: Creating Command Queue. (clCreateCommandQueueWithProperties) returned %d\n", err );
		exit(EXIT_FAILURE);
    	}

	hardware.platform = platform;
	hardware.device = device;
	hardware.queue = queue;
	hardware.context = ctx;

 	char device_name[1024];
 	char device_vend[1024];
 	char device_driver[1024];
	cl_uint CUs;
	cl_ulong maxMemAllocSize;

	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vend), &device_vend, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_driver), &device_driver, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CUs, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAllocSize, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	sd.maxmalloc = (uint64_t)maxMemAllocSize;

	fprintf(stderr, "GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_name, device_vend, device_driver, CUs);
	if(boinc_is_standalone()){
		printf("GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_name, device_vend, device_driver, CUs);
	}

	// check vendor and normalize compute units
	// kernel size will be determined by profiling so this doesn't have to be accurate.
	sd.computeunits = (uint32_t)CUs;
	char intel_s[] = "Intel";
	char arc_s[] = "Arc";
	char nvidia_s[] = "NVIDIA";	

	if(strstr((char*)device_vend, (char*)nvidia_s) != NULL){
#ifdef _WIN32
		// pascal or newer gpu on windows 10,11 allows long kernel runtimes without screen refresh issues

		float winVer = (float)getSysOpType();

		if(winVer >= 10.0f && !sd.compute){

		 	cl_uint ccmajor;
			err = clGetDeviceInfo(hardware.device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(ccmajor), &ccmajor, NULL);
			if ( err != CL_SUCCESS ) {
				printf( "Error checking device compute capability\n" );
				fprintf(stderr, "Error checking device compute capability\n");
				exit(EXIT_FAILURE);
			}

			if(ccmajor >= 6){
				sd.compute = true;
			}
		}

#else
		// linux
		// list of popular gpus without video output
		char dc0[] = "P100";
		char dc1[] = "V100";
		char dc2[] = "T4";
		char dc3[] = "A100";
		char dc4[] = "L4";
		char dc5[] = "H100";
		char dc6[] = "H200";
		char dc7[] = "B100";
		char dc8[] = "B200";

		if(	strstr((char*)device_name, (char*)dc0) != NULL
			|| strstr((char*)device_name, (char*)dc1) != NULL
			|| strstr((char*)device_name, (char*)dc2) != NULL
			|| strstr((char*)device_name, (char*)dc3) != NULL
			|| strstr((char*)device_name, (char*)dc4) != NULL
			|| strstr((char*)device_name, (char*)dc5) != NULL
			|| strstr((char*)device_name, (char*)dc6) != NULL
			|| strstr((char*)device_name, (char*)dc7) != NULL
			|| strstr((char*)device_name, (char*)dc8) != NULL){
			sd.compute = true;
		}

#endif
	}
	// Intel
	else if( strstr((char*)device_vend, (char*)intel_s) != NULL ){
		if( strstr((char*)device_name, (char*)arc_s) != NULL ){
			sd.computeunits /= 10;
		}
		else{
			sd.computeunits /= 20;
	                fprintf(stderr,"Detected Intel integrated graphics\n");	
		}
	}
	// AMD
        else{
		sd.computeunits /= 2;
        }

	if(!sd.computeunits) sd.computeunits++;
	
	if(sd.test){
		run_test(hardware, st, sd);
	}
	else{
		cl_sieve(hardware, st, sd);
	}

        sclReleaseClHard(hardware);

	boinc_finish(EXIT_SUCCESS);

	return 0; 
} 

