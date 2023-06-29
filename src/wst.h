#include <complex>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <armadillo>

#include "pocketfft_hdronly.h"

typedef std::vector<double> vec1D;
typedef std::vector<std::complex<double>> vec1D_complex;
typedef std::vector<vec1D_complex> vec2D_complex;
typedef std::vector<vec2D_complex> vec3D_complex;
typedef std::vector<vec1D> vec2D;
typedef std::vector<vec2D> vec3D;
typedef std::vector<vec2D_complex> vec3D_complex;
typedef std::vector<vec3D_complex> vec4D_complex;
typedef std::vector<vec3D_complex> vec4D_complex;
typedef std::vector<vec4D_complex> vec5D_complex;

struct Psi
{
    vec3D_complex levels;
    int j;
    int theta;
};

struct Phi
{
    vec3D_complex levels;
    int j;
};

struct filterBank
{
    Phi phi;
    std::vector<Psi> psi_filters;
};

struct outputs
{
    vec2D_complex coef;
    std::vector<int> j;
    std::vector<int> n;
    std::vector<int> theta;
};

vec2D_complex gabor_wavelet_2d(int M, int N, double sigma, double theta, double xi, double slant, double offset);


vec2D_complex morlet_wavelet_2d(int M, int N, double sigma, double theta, double xi, double scale);


void print_1d(std::vector<int> vec);


void print_3d(vec3D vec);


void print_2d(vec2D vec);


void print_complex_2d(vec2D_complex vec);


void print_complex_3d(vec3D_complex vec);


vec2D_complex modulus(const vec2D_complex &in);


vec2D_complex convert_to_complex(const vec2D &in);


vec2D_complex periodize_filter_fft(vec2D_complex x, int res);


vec1D_complex dft(const vec1D_complex &in);


static vec1D_complex fft(const vec1D_complex &in);


vec1D_complex ifft(const vec1D_complex &in);


vec2D_complex fft2D(const vec2D_complex &in);


vec2D_complex ifft2D(const vec2D_complex &in);


filterBank filter_bank(int M, int N, int J, int L = 8);

filterBank filter_bank_fast(int M, int N, int J, int L = 8);

std::tuple<vec2D, vec1D> load(const std::string& images_path, int i = -1);


vec3D pad(vec1D digit, int J);


vec2D_complex cdgmm_arma(vec2D_complex raw, vec2D_complex filter);


vec2D_complex cdgmm(vec2D_complex raw, vec2D_complex filter);


vec2D_complex subsample_fourier(vec2D_complex raw, int size);


vec2D_complex unpad(vec2D_complex subsampled);


vec2D_complex fft2D_pocket(const vec2D_complex &in);

vec2D_complex ifft2D_pocket(const vec2D_complex &in);

std::vector<outputs> scatter_fast(vec2D signal, vec1D digit, filterBank filter_data, int J);

std::vector<outputs> scatter_fast(vec2D signal, vec1D digit);

std::vector<outputs> scatter(vec2D signal, vec1D digit);

std::vector<outputs> scatter(vec2D signal, vec1D digit, filterBank filter_data, int J);

// Softmax function
vec1D softmax(vec1D x);

// Prediction function
vec1D predict(const vec1D &x, const vec2D &coef, const vec1D &intercept);
