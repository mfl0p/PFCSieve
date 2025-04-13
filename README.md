# PFCSieve

PFCSieve by Bryan Little

A BOINC enabled OpenCL standalone sieve for factors of factorial, primorial, and compositorial prime candidates of the form n!+-1, n#+-1, and n!/#+-1

Using OpenMP for multithreaded factor verification on CPU.

With contributions by
* Yves Gallot
* Mark Rodenkirch
* Kim Walisch

## Requirements

* OpenCL v1.1
* 64 bit operating system

## How it works

1. Search parameters are given on the command line.
2. A small group of sieve primes are generated on the GPU.
3. The group of primes are tested for factors in the N range specified.
4. Repeat #2-3 until checkpoint.  Gather factors and checksum data from GPU.
5. Report any factors in factors.txt, along with a checksum at the end.
6. Checksum can be used to compare results in a BOINC quorum.

## Running the program
```
command line options
* -!	Use factorial mode
* -#	Use primorial mode
* -c	Use compositorial mode
*		Note: -! -c can be used together to find factors of both at the same time.
* -n #	Start primorial n#+-1, factorial n!+-1, or compositorial n!/#+-1
* -N #	End primorial N#+-1, factorial N!+-1, or compositorial N!/#+-1
* 		N range is 101 <= -n < -N < 2^31, [-n, -N) exclusive
* -p #	Starting prime factor p
* -P #	End prime factor P
* 		P range is 3 <= -p < -P < 2^64, [-p, -P) exclusive
* 		Note for primorial and factorial there are no factors when p <= n
* 		Note N!+-1, N#+-1, and N!/#+-1 are not divisible by 2.
* -v #	Optional, specify the number of CPU threads used to verify factors.  Default is 2, max is 128.
* -s 	Perform self test to verify proper operation of the program with the current GPU.
* -h	Print help

Program gets the OpenCL GPU device index from BOINC.  To run stand-alone, the program will
default to GPU 0 unless an init_data.xml is in the same directory with the format:

<app_init_data>
<gpu_type>NVIDIA</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>

or

<app_init_data>
<gpu_type>ATI</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>
```

## Related Links
* [Yves Gallot on GitHub](https://github.com/galloty)
* [primesieve by Kim Walisch](https://github.com/kimwalisch/primesieve)
* [Mark Rodenkirch on SourceForge](https://sourceforge.net/projects/mtsieve/)
