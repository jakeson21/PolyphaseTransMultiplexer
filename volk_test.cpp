#include <iostream>
#include <vector>
#include <cmath>
#include <volk/volk.h>
#include <fftw3.h>
#include <boost/date_time.hpp>

class Timer
{
public:
    static void start()
    {
        // Get the current time before executing Task
        Timer::t0 = boost::posix_time::microsec_clock::local_time();
    }
    static void stop()
    {
        // Get the current time after executing Task
        Timer::tend = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration dur = tend - t0;
        // Getting the Time Difference in Total Nano Seconds only
        std::cout << "Time elapsed in usec:  " << dur.total_microseconds() << std::endl;        
    }
    
protected:
    Timer() = delete;
    
    static boost::posix_time::ptime t0;
    static boost::posix_time::ptime tend;
};

boost::posix_time::ptime Timer::t0;
boost::posix_time::ptime Timer::tend;



int main(int argc, char **argv) {
    std::cout << "Hello, world!" << std::endl;
    std::cout << "don't forget to run volk_profile" << std::endl;
    
    std::cout << "Alignment is: " << volk_get_alignment() << std::endl;
    
    {
        int N = 62500000;
        unsigned int alignment = volk_get_alignment();
        lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
        lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
        for(unsigned int ii = 0; ii < N; ++ii){
            // Generate a tone at f=0.3
            float real = std::cos(0.3f * (float)ii);
            float imag = std::sin(0.3f * (float)ii);
            in[ii] = lv_cmake(real, imag);
        }
        // The oscillator rotates at f=0.1
        float frequency = 0.1f;
        lv_32fc_t phase_increment = lv_cmake(std::cos(frequency), std::sin(frequency));
        lv_32fc_t phase= lv_cmake(1.f, 0.0f); // start at 1 (0 rad phase)
        
        // rotate so the output is a tone at f=0.4
        std::cout << "Running volk_32fc_s32fc_x2_rotator_32fc for length " << N << " vectors" << std::endl;
        Timer::start();
        volk_32fc_s32fc_x2_rotator_32fc(out, in, phase_increment, &phase, N);
        Timer::stop();
        
        volk_free(in);
        volk_free(out);
    }

    
    {
        int N = 62500000;
        unsigned int alignment = volk_get_alignment();
        lv_32fc_t* increasing = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
        lv_32fc_t* decreasing = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
        lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
        for(unsigned int ii = 0; ii < N; ++ii){
            increasing[ii] = (lv_32fc_t)ii;
            decreasing[ii] = 10.f - (lv_32fc_t)ii;
        }
        std::cout << "Running volk_32fc_x2_add_32fc for length " << N << " vectors" << std::endl;
        Timer::start();
        volk_32fc_x2_add_32fc(out, increasing, decreasing, N);
        Timer::stop();
        
        volk_free(increasing);
        volk_free(decreasing);
        volk_free(out);
    }

    {
        fftw_complex *in, *out;
        fftw_plan p;
        
        int64_t FS = 62500000;
        size_t blocks = 100;
        size_t N = FS/blocks;
        N = std::pow(2, std::ceil(std::log2(N)));        
        
        if(!fftw_init_threads())
        {
            exit(1);
        }
        int nthreads = 8;
        fftw_plan_with_nthreads(nthreads);
        
        in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        
        std::cout << "Running fftw_execute " << blocks << " times for NFFT of " << N << std::endl;
        Timer::start();
        for (size_t n=0; n<blocks; n++)
            fftw_execute(p); /* repeat as needed */
        Timer::stop();
        
        fftw_destroy_plan(p);
        fftw_free(in); fftw_free(out);
        fftw_cleanup_threads();
    }
    
    
    return 0;    
}
