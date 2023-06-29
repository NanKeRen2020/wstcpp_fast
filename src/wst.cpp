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

#include "omp.h"
#include "pocketfft_hdronly.h"
#include "wst.h"

vec2D_complex gabor_wavelet_2d(int M, int N, double sigma, double theta, double xi, double slant, double offset)
{
    // Initialize the 2D Gabor matrix
    vec2D_complex gab(M, vec1D_complex(N, std::complex<double>(0, 0)));

    // Define rotation matrices
    double R[2][2] = {{std::cos(theta), -std::sin(theta)}, {std::sin(theta), std::cos(theta)}};
    double R_inv[2][2] = {{std::cos(theta), std::sin(theta)}, {-std::sin(theta), std::cos(theta)}};
    double D[2][2] = {{1, 0}, {0, slant * slant}};
    double curv[2][2];

    // dot product of D and R_inv
    double curv1[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            double sum = 0;
            for (int k = 0; k < 2; k++)
                sum += D[i][k] * R_inv[k][j];
            curv1[i][j] = sum;
        }

    // dot product of R and curv
    double curv2[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            double sum = 0;
            for (int k = 0; k < 2; k++)
                sum += R[i][k] * curv1[k][j];
            curv2[i][j] = sum;
        }

    // divide by 2 * sigma * sigma
    double sigma_sq = 2 * sigma * sigma; // Calculation done outside the loop for efficiency.
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            curv2[i][j] /= sigma_sq;

    // copy curv2 to curv
    std::copy(&curv2[0][0], &curv2[0][0] + 4, &curv[0][0]);

    // Compute Gabor filter
    for (int ex : {-2, -1, 0, 1, 2})
        for (int ey : {-2, -1, 0, 1, 2})
        {

            vec2D_complex xx(M, vec1D_complex(N, 0));
            vec2D_complex yy(M, vec1D_complex(N, 0));

            for (int i = offset + ex * M; i < offset + M + ex * M; i++)
                for (int j = offset + ey * N; j < offset + N + ey * N; j++)
                {
                    xx[i - offset - ex * M][j - offset - ey * N] = i;
                    yy[i - offset - ex * M][j - offset - ey * N] = j;
                }

            vec2D_complex arg(M, vec1D_complex(N, 0));
            for (int i = 0; i < xx.size(); i++)
            {
                for (int j = 0; j < xx[i].size(); j++)
                {

                    std::complex<double> left = -(curv[0][0] * xx[i][j] * xx[i][j] + (curv[0][1] + curv[1][0]) * xx[i][j] * yy[i][j] + curv[1][1] * yy[i][j] * yy[i][j]);
                    std::complex<double> right = std::complex<double>(0, 1) * (xx[i][j] * xi * std::cos(theta) + yy[i][j] * xi * std::sin(theta));

                    std::complex<double> val = left + right;
                    arg[i][j] = val;
                }
            }

            for (int i = 0; i < arg.size(); i++)
                for (int j = 0; j < arg[i].size(); j++)
                    gab[i][j] +=
                        std::complex<double>(std::round(std::exp(arg[i][j]).real() * 1e44) / 1e44, std::round(std::exp(arg[i][j]).imag() * 1e44) / 1e44);
        }

    // Normalize the Gabor matrix
    double norm_factor = (2 * 3.1415 * sigma * sigma / slant);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            gab[i][j] /= norm_factor;

    // only keep real part
    vec2D gab_real(M, vec1D(N, 0));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; ++j)
            gab_real[i][j] = gab[i][j].real();

    return gab;
}

vec2D_complex morlet_wavelet_2d(int M, int N, double sigma, double theta, double xi, double scale)
{
    vec2D_complex wv = gabor_wavelet_2d(M, N, sigma, theta, xi, 0.5, 0);
    vec2D_complex wv_modulus = gabor_wavelet_2d(M, N, sigma, theta, 0, 0.5, 0);

    // sum all values in wv
    double wv_sum = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; ++n)
            wv_sum += wv[m][n].real();

    // sum all values in wv_modulus
    double wv_modulus_sum = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; ++n)
            wv_modulus_sum += wv_modulus[m][n].real();

    // K is sum of wv divided by sum of wv_modulus
    double K = wv_sum / wv_modulus_sum;
    vec2D_complex mor;
    for (int m = 0; m < M; m++)
    {
        vec1D_complex row;
        for (int n = 0; n < N; n++)
        {
            row.push_back(wv[m][n] - K * wv_modulus[m][n]);
        }
        mor.push_back(row);
    }

    return mor;
}

void print_1d(std::vector<int> vec)
{
    for (const auto &element : vec)
        std::cout << std::endl;
}

void print_3d(vec3D vec)
{
    for (const auto &matrix : vec)
        for (const auto &vec : matrix)
        {
            for (const auto &element : vec)
                std::cout << element << " ";
            std::cout << std::endl;
        }
}

void print_2d(vec2D vec)
{
    for (const auto &vec : vec)
        for (const auto &element : vec)
            std::cout << element << " ";
    std::cout << std::endl;
}

void print_complex_2d(vec2D_complex vec)
{
    for (const auto &vec : vec)
    {
        for (const auto &element : vec)
            std::cout << std::fixed << std::setw(7) << std::setprecision(4) << element.real() << " ";
        std::cout << std::endl;
    }
}

void print_complex_3d(vec3D_complex vec)
{
    for (const auto &matrix : vec)
        for (const auto &vec : matrix)
        {
            for (const auto &element : vec)
                std::cout << element.real() << " ";
            std::cout << std::endl;
        }
}

vec2D_complex modulus(const vec2D_complex &in)
{
    vec2D_complex out;
    for (const auto &row : in)
    {
        vec1D_complex complex_row;
        for (const auto &val : row)
            complex_row.push_back(std::abs(val));
        out.push_back(complex_row);
    }
    return out;
}

vec2D_complex convert_to_complex(const vec2D &in)
{
    vec2D_complex out;
    for (const auto &row : in)
    {
        vec1D_complex complex_row;
        for (double val : row)
            complex_row.push_back(std::complex<double>(val, 0.0));
        out.push_back(complex_row);
    }
    return out;
}

vec2D_complex periodize_filter_fft(vec2D_complex x, int res)
{
    int M = x.size();
    int N = x[0].size();
    vec2D_complex crop(M / pow(2, res), vec1D_complex(N / pow(2, res), 0));

    std::vector<std::vector<float>> mask(M, std::vector<float>(N, 1.0f));
    int len_x = int(M * (1 - pow(2, -res)));
    int start_x = int(M * pow(2, -res - 1));
    int len_y = int(N * (1 - pow(2, -res)));
    int start_y = int(N * pow(2, -res - 1));

    for (int i = start_x; i < start_x + len_x; ++i)
        for (int j = 0; j < N; ++j)
            mask[i][j] = 0;

    for (int i = 0; i < M; ++i)
        for (int j = start_y; j < start_y + len_y; ++j)
            mask[i][j] = 0;

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            x[i][j] *= mask[i][j];

    for (int k = 0; k < M / pow(2, res); ++k)
        for (int l = 0; l < N / pow(2, res); ++l)
            for (int i = 0; i < pow(2, res); ++i)
                for (int j = 0; j < pow(2, res); ++j)
                    crop[k][l] += x[k + i * int(M / pow(2, res))][l + j * int(N / pow(2, res))];

    return crop;
}

vec1D_complex dft(const vec1D_complex &in)
{
    int N = in.size();
    vec1D_complex out(N);
    for (int k = 0; k < N; k++)
    {
        std::complex<double> sum = 0;
        for (int n = 0; n < N; n++)
        {
            double angle = 2 * M_PI * k * n / N;
            std::complex<double> c(std::cos(angle), -std::sin(angle));
            sum += in[n] * c;
        }
        out[k] = sum;
    }
    return out;
}

static vec1D_complex fft(const vec1D_complex &in)
{
    int N = in.size();
    vec1D_complex out(N);
    if (N == 1)
    {
        out[0] = in[0];
        return out;
    }
    if (N % 2 == 1)
        return dft(in); // Call a 1D version of DFT if N is not a power of 2

    vec1D_complex even(N / 2), odd(N / 2);
    for (int i = 0; i < N; i++)
    {
        if (i % 2 == 0)
            even[i / 2] = in[i];
        else
            odd[i / 2] = in[i];
    }
    vec1D_complex even_fft = fft(even);
    vec1D_complex odd_fft = fft(odd);
    for (int k = 0; k < N / 2; k++)
    {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd_fft[k];
        out[k] = even_fft[k] + t;
        out[k + N / 2] = even_fft[k] - t;
    }
    return out;
}

vec1D_complex ifft(const vec1D_complex &in)
{
    int N = in.size();
    vec1D_complex out(N);
    vec1D_complex conjugate_in(N);
    for (int i = 0; i < N; i++)
        conjugate_in[i] = std::conj(in[i]);

    vec1D_complex conjugate_out = fft(conjugate_in);
    for (int i = 0; i < N; i++)
        out[i] = std::conj(conjugate_out[i]) / static_cast<double>(N);

    return out;
}

vec2D_complex fft2D(const vec2D_complex &in)
{
    int H = in.size();
    int W = in[0].size();
    vec2D_complex out(H, vec1D_complex(W));

    // Apply FFT on rows
    for (int i = 0; i < H; i++)
        out[i] = fft(in[i]);

    // Apply FFT on columns
    for (int j = 0; j < W; j++)
    {
        vec1D_complex col_in_complex(H);
        for (int i = 0; i < H; i++)
            col_in_complex[i] = out[i][j];

        vec1D_complex col_out = fft(col_in_complex);
        for (int i = 0; i < H; i++)
            out[i][j] = col_out[i];
    }

    return out;
}

vec2D_complex ifft2D(const vec2D_complex &in)
{
    int H = in.size();
    int W = in[0].size();
    vec2D_complex out(H, vec1D_complex(W));

    // Apply FFT on rows
    for (int i = 0; i < H; i++)
        out[i] = ifft(in[i]);

    // Apply FFT on columns
    for (int j = 0; j < W; j++)
    {
        vec1D_complex col_in_complex(H);
        for (int i = 0; i < H; i++)
            col_in_complex[i] = out[i][j];

        vec1D_complex col_out = ifft(col_in_complex);
        for (int i = 0; i < H; i++)
            out[i][j] = col_out[i];
    }

    return out;
}

filterBank filter_bank(int M, int N, int J, int L)
{
    filterBank fb;

    Phi phi_filters = {{}, J};
    std::vector<Psi> psi_filters;

    for (int j = 0; j < J; ++j)
        for (int theta = 0; theta < L; ++theta)
        {
            Psi psi_filter = {{}, j, theta};
            vec1D_complex levels = {};
            vec2D_complex psi_signal = morlet_wavelet_2d(
                M,
                N,
                0.8 * std::pow(2, j),
                (L - L / 2 - 1 - theta) * M_PI / L,
                3.0 / 4.0 * M_PI / std::pow(2, j),
                4.0 / L);

            vec2D_complex psi_signal_fourier = fft2D(psi_signal);

            for (int i = 0; i < psi_signal_fourier.size(); ++i)
                for (int j = 0; j < psi_signal_fourier[0].size(); ++j)
                    psi_signal_fourier[i][j] = psi_signal_fourier[i][j].real();

            for (int res = 0; res < std::min(j + 1, std::max(J - 1, 1)); ++res)
            {

                vec2D_complex crop = periodize_filter_fft(psi_signal_fourier, res);
                psi_filter.levels.push_back(crop);
            }

            psi_filters.push_back(psi_filter);
        }

    vec2D_complex phi_signal = gabor_wavelet_2d(M, N, 0.8 * pow(2, J - 1), 0, 0, 1.0, 0);
    vec2D_complex phi_signal_fourier = fft2D(phi_signal);

    // only keep real part
    for (int i = 0; i < phi_signal_fourier.size(); ++i)
        for (int j = 0; j < phi_signal_fourier[0].size(); ++j)
            phi_signal_fourier[i][j] = phi_signal_fourier[i][j].real();

    for (int i = 0; i < J; ++i)
        phi_filters.levels.push_back(periodize_filter_fft(phi_signal_fourier, i));

    fb.phi = phi_filters;
    fb.psi_filters = psi_filters;

    return fb;
}

filterBank filter_bank_fast(int M, int N, int J, int L)
{
    filterBank fb;

    Phi phi_filters = {{}, J};
    std::vector<Psi> psi_filters;

    for (int j = 0; j < J; ++j)
        for (int theta = 0; theta < L; ++theta)
        {
            auto start = std::chrono::high_resolution_clock::now();
            Psi psi_filter = {{}, j, theta};
            vec1D_complex levels = {};
            vec2D_complex psi_signal = morlet_wavelet_2d(
                M,
                N,
                0.8 * std::pow(2, j),
                (L - L / 2 - 1 - theta) * M_PI / L,
                3.0 / 4.0 * M_PI / std::pow(2, j),
                4.0 / L);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "morlet_wavelet_2d time: " << diff.count() << std::endl;

            vec2D_complex psi_signal_fourier = fft2D_pocket(psi_signal);

            for (int i = 0; i < psi_signal_fourier.size(); ++i)
                for (int j = 0; j < psi_signal_fourier[0].size(); ++j)
                    psi_signal_fourier[i][j] = psi_signal_fourier[i][j].real();

            for (int res = 0; res < std::min(j + 1, std::max(J - 1, 1)); ++res)
            {

                vec2D_complex crop = periodize_filter_fft(psi_signal_fourier, res);
                psi_filter.levels.push_back(crop);
            }

            psi_filters.push_back(psi_filter);
        }

    auto start = std::chrono::high_resolution_clock::now();
    vec2D_complex phi_signal = gabor_wavelet_2d(M, N, 0.8 * pow(2, J - 1), 0, 0, 1.0, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "gabor_wavelet_2d time: " << diff.count() << std::endl;

    vec2D_complex phi_signal_fourier = fft2D_pocket(phi_signal);

    // only keep real part
    for (int i = 0; i < phi_signal_fourier.size(); ++i)
        for (int j = 0; j < phi_signal_fourier[0].size(); ++j)
            phi_signal_fourier[i][j] = phi_signal_fourier[i][j].real();

    for (int i = 0; i < J; ++i)
        phi_filters.levels.push_back(periodize_filter_fft(phi_signal_fourier, i));

    fb.phi = phi_filters;
    fb.psi_filters = psi_filters;

    return fb;
}

std::tuple<vec2D, vec1D> load(const std::string& images_path, int i)
{

    uint8_t buf[784];
    vec1D digit;

    //auto fin = std::ifstream("/home/seeking/test_projects/nkrcv/wstcpp/data/t10k-images-idx3-ubyte", std::ios::binary);
    auto fin = std::ifstream(images_path, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "failed to open digits file\n");
        // return 0;
    }
    srand(time(NULL));

    // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
    if (i < 0)
    fin.seekg(16 + 784 * (rand() % 10000));
    else
    {
        fin.seekg(16 + 784 * (i % 10000));

    }
    
    // // get first digit
    // fin.seekg(16 + 784);

    fin.read((char *)&buf, sizeof(buf));

    // render the digit in ASCII
    {
        digit.resize(sizeof(buf));
        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; col++)
            {
                fprintf(stderr, "%c ", (float)buf[row * 28 + col] > 230 ? '*' : '_');
                digit[row * 28 + col] = ((float)buf[row * 28 + col]) / 255.0;
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }

    // put the digit in a 2D vector
    vec2D signal(28, vec1D(28));
    {
        vec1D time(784);
        std::iota(time.begin(), time.end(), 0);

        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; col++)
            {
                signal[row][col] = digit[row * 28 + col];
            }
        }
    }

    return std::make_tuple(signal, digit);
}

vec3D pad(vec1D digit, int J)
{
    // pad the signal to 36x36 padding should be applied equally on both sides
    vec2D padded_signal(36, vec1D(36));
    {
        int padding = 4;
        if (padding >= 28)
        {
            throw std::invalid_argument("Indefinite padding size (larger than tensor).");
        }

        for (int row = 0; row < 36; row++)
        {
            for (int col = 0; col < 36; col++)
            {
                // Calculate the corresponding indices in the original image
                int orig_row = row - padding;
                int orig_col = col - padding;

                // Reflect the indices if they're out of bounds
                if (orig_row < 0)
                    orig_row = -orig_row; // Reflect on the top edge
                else if (orig_row >= 28)
                    orig_row = 2 * 28 - orig_row - 2; // Reflect on the bottom edge

                if (orig_col < 0)
                    orig_col = -orig_col; // Reflect on the left edge
                else if (orig_col >= 28)
                    orig_col = 2 * 28 - orig_col - 2; // Reflect on the right edge

                // After reflection, if the indices are still out of bounds (this might happen when padding size >= image size), clamp them
                orig_row = std::max(0, std::min(orig_row, 27)); // Clamp between 0 and 27
                orig_col = std::max(0, std::min(orig_col, 27)); // Clamp between 0 and 27

                // Get the pixel value from the original image, or 0 if it's out of bounds
                if (orig_row >= 0 && orig_row < 28 && orig_col >= 0 && orig_col < 28)
                    padded_signal[row][col] = digit[orig_row * 28 + orig_col];
                else
                    padded_signal[row][col] = 0;
            }
        }
    }

    // convert padded_signal into a 3D vector 1x36x36
    vec3D padded_signal_3d(1, vec2D(36, vec1D(36)));
    for (int row = 0; row < 36; row++)
        for (int col = 0; col < 36; col++)
            padded_signal_3d[0][row][col] = padded_signal[row][col];

    return padded_signal_3d;
}

vec2D_complex cdgmm_arma(vec2D_complex raw, vec2D_complex filter)
{
    arma::Mat<arma::cx_double> raw_arma(raw.size(), raw[0].size());
    for (int i = 0; i < raw.size(); ++i)
    {
        for (int j = 0; j < raw[0].size(); ++j)
             raw_arma(i, j) = raw[i][j];
    }
    arma::Mat<arma::cx_double> filter_arma(filter.size(), filter.size());
    for (int i = 0; i < filter.size(); ++i)
    {
        for (int j = 0; j < filter[0].size(); ++j)
             filter_arma(i, j) = filter[i][j];
    }
    arma::Mat<arma::cx_double> r = raw_arma%filter_arma;
    vec2D_complex result = raw;
     for (int i = 0; i < raw.size(); ++i)
    {
        for (int j = 0; j < raw[0].size(); ++j)
           result[i][j] = r(i, j);
    }   
    return result;
}

vec2D_complex cdgmm(vec2D_complex raw, vec2D_complex filter)
{
    int M = raw.size();
    int N = raw[0].size();

    // A * B; point wise multiplication between raw and filter
    vec2D_complex filtered(M, vec1D_complex(N));
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
            filtered[row][col] = raw[row][col] * filter[row][col];

    return filtered;
}

vec2D_complex subsample_fourier(vec2D_complex raw, int size)
{

    int M = raw.size();
    int N = raw[0].size();

    //assert(M == N);

    int n = M / size;

    // Initialize the 5D vector
    vec4D_complex y(size, vec3D_complex(n, vec2D_complex(size, vec1D_complex(n))));
    for (int j = 0; j < size; j++)
        for (int k = 0; k < n; k++)
            for (int l = 0; l < size; l++)
                for (int m = 0; m < n; m++)
                {
                    int x_row = j * n + k;
                    int x_col = l * n + m;
                    y[j][k][l][m] = raw[x_row][x_col];
                }

    vec2D_complex out(n, vec1D_complex(n));
    for (int k = 0; k < n; k++)
        for (int m = 0; m < n; m++)
        {
            double sum = 0.0;
            double im_sum = 0.0;
            for (int j = 0; j < size; j++)
                for (int l = 0; l < size; l++)
                {
                    // Add up all values in the 2nd and 4th dimensions
                    sum += y[j][k][l][m].real();
                    im_sum += y[j][k][l][m].imag();
                }
            // Divide by the number of elements to get the average
            out[k][m] = std::complex<double>(sum / std::pow(size, 2), im_sum / std::pow(size, 2));
        }
    return out;
}



vec2D_complex unpad(vec2D_complex subsampled)
{
    int rows = 7;
    int cols = 7;
    vec2D_complex unpadded(rows, vec1D_complex(cols));
    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            unpadded[row][col] = subsampled[row + 1][col + 1];
    return unpadded;
}

vec2D_complex fft2D_pocket(const vec2D_complex &in)
{
    pocketfft::shape_t shape{in.size(), in[0].size()};
    pocketfft::stride_t stride(shape.size());
    size_t tmp=sizeof(std::complex<double>);
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        stride[i]=tmp;
        tmp*=shape[i];
    }
    size_t ndata=1;
    for (size_t i=0; i<shape.size(); ++i)
      ndata*=shape[i];
    std::vector<std::complex<double>> datas(ndata);
    for (int i = 0; i < shape[0]; ++i)
    {
        for (int j = 0; j < shape[1]; ++j)
        {
            datas[j + shape[1]*i] = in[j][i];

        }
    }
    pocketfft::shape_t axes;
    for (size_t i=0; i<shape.size(); ++i)
       axes.push_back(i);
    auto res = datas;
    auto start = std::chrono::high_resolution_clock::now();
    pocketfft::c2c(shape, stride, stride, axes, pocketfft::detail::FORWARD,
                   datas.data(), res.data(), 1.);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    //std::cout << "c2c time: " << diff.count() << std::endl;
    vec2D_complex out = in;
    for (int i = 0; i < shape[0]; ++i)
    {
        for (int j = 0; j < shape[1]; ++j)
            out[i][j] = res[j*shape[1] + i];
    }

    return out;
}

vec2D_complex ifft2D_pocket(const vec2D_complex &in)
{
    pocketfft::shape_t shape{in.size(), in[0].size()};
    pocketfft::stride_t stride(shape.size());
    size_t tmp=sizeof(std::complex<double>);
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        stride[i]=tmp;
        tmp*=shape[i];
    }
    size_t ndata=1;
    for (size_t i=0; i<shape.size(); ++i)
      ndata*=shape[i];
    std::vector<std::complex<double>> datas(ndata);
    for (int i = 0; i < shape[0]; ++i)
    {
        for (int j = 0; j < shape[1]; ++j)
        {
            datas[j + shape[1]*i] = in[j][i];

        }
    }
    pocketfft::shape_t axes;
    for (size_t i=0; i<shape.size(); ++i)
       axes.push_back(i);
    auto res = datas;
    auto start = std::chrono::high_resolution_clock::now();
    pocketfft::c2c(shape, stride, stride, axes, pocketfft::detail::BACKWARD,
                   datas.data(), res.data(), 1.);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    //std::cout << "c2c time: " << diff.count() << std::endl;
    vec2D_complex out = in;
    for (int i = 0; i < shape[0]; ++i)
    {
        for (int j = 0; j < shape[1]; ++j)
            out[i][j] = res[j*shape[1] + i]/std::complex<double>(ndata, 0);
    }

    return out;
}

std::vector<outputs> scatter_fast(vec2D signal, vec1D digit)
{
    auto start = std::chrono::high_resolution_clock::now();
    /*
        specify the filter bank parameters
    */
    int M, N, J, L;
    M = N = 36;
    J = 2;
    L = 8;

    filterBank filter_data = filter_bank_fast(M, N, J, L);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "scatter_fast filter_bank time: " << diff.count() << std::endl;
    
    /*
        specify the input signal
    */
    vec3D U_r = pad(digit, 1);
    vec2D_complex U_0_c(36, vec1D_complex(36));
    start = std::chrono::high_resolution_clock::now();
    U_0_c = fft2D_pocket(convert_to_complex(U_r[0]));
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "fft2D_pocket time: " << diff.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vec2D_complex U_1_c = cdgmm_arma(U_0_c, filter_data.phi.levels[0]);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "cdgmm_arma time: " << diff.count() << std::endl;

    vec2D_complex subsample = subsample_fourier(U_1_c, std::pow(2, J));
    start = std::chrono::high_resolution_clock::now();
    vec2D_complex S_0 = ifft2D_pocket(subsample);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "ifft2D_pocket time: " << diff.count() << std::endl;

    vec2D_complex S_0_unpadded = unpad(S_0);

    std::vector<outputs> out_S_0;
    std::vector<outputs> out_S_1;
    std::vector<outputs> out_S_2;

    out_S_0.push_back({S_0_unpadded, {}, {}, {}});
    int max_order = 2;
    start = std::chrono::high_resolution_clock::now();
    /*
        calculate the scattering coefficients
    */
    for (int n1 = 0; n1 < filter_data.psi_filters.size(); n1++)
    {
        int j1 = filter_data.psi_filters[n1].j;
        int theta1 = filter_data.psi_filters[n1].theta;

        vec2D_complex psi_f = filter_data.psi_filters[n1].levels[0];
        vec2D_complex phi_f = filter_data.phi.levels[j1];

        /*
            calculate the first order scattering coefficients
        */
        vec2D_complex U_1_c = cdgmm_arma(U_0_c, psi_f);
        if (j1 > 0)
            U_1_c = subsample_fourier(U_1_c, std::pow(2, j1));

        // inverse modulus and fft
        U_1_c = ifft2D_pocket(U_1_c);
        U_1_c = modulus(U_1_c);
        U_1_c = fft2D_pocket(U_1_c);

        // low pass filter
        vec2D_complex S_1_c = cdgmm_arma(U_1_c, phi_f);
        S_1_c = subsample_fourier(S_1_c, std::pow(2, (J - j1)));

        vec2D_complex S_1_r = ifft2D_pocket(S_1_c);
        S_1_r = unpad(S_1_r);

        // store the scattering coefficients
        out_S_1.push_back({S_1_r, {j1}, {n1}, {theta1}});

        if (max_order < 2)
            continue;

        /*
            calculate the second order scattering coefficients
        */
        for (int n2 = 0; n2 < filter_data.psi_filters.size(); n2++)
        {
            int j2 = filter_data.psi_filters[n2].j;
            int theta2 = filter_data.psi_filters[n2].theta;

            if (j2 <= j1)
                continue;

            vec2D_complex psi_f2 = filter_data.psi_filters[n2].levels[j1];
            vec2D_complex phi_f2 = filter_data.phi.levels[j2];

            vec2D_complex U_2_c = cdgmm_arma(U_1_c, psi_f2);
            U_2_c = subsample_fourier(U_2_c, std::pow(2, (j2 - j1)));

            // inverse modulus and fft
            U_2_c = ifft2D_pocket(U_2_c);
            U_2_c = modulus(U_2_c);
            U_2_c = fft2D_pocket(U_2_c);

            // low pass filter
            vec2D_complex S_2_c = cdgmm_arma(U_2_c, phi_f2);
            S_2_c = subsample_fourier(S_2_c, std::pow(2, (J - j2)));

            vec2D_complex S_2_r = ifft2D_pocket(S_2_c);
            S_2_r = unpad(S_2_r);

            // store the scattering coefficients
            out_S_2.push_back({S_2_r, {j1, j2}, {n1, n2}, {theta1, theta2}});
        }
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "scatter_fast loop time: " << diff.count() << std::endl;


    // join all the lists
    std::vector<outputs> out_S;
    out_S.insert(out_S.end(), out_S_0.begin(), out_S_0.end());
    out_S.insert(out_S.end(), out_S_1.begin(), out_S_1.end());
    out_S.insert(out_S.end(), out_S_2.begin(), out_S_2.end());

    return out_S;
}


std::vector<outputs> scatter_fast(vec2D signal, vec1D digit, filterBank filter_data, int J)
{    
    /*
        specify the input signal
    */
    vec3D U_r = pad(digit, 1);
    vec2D_complex U_0_c(36, vec1D_complex(36));
    auto start = std::chrono::high_resolution_clock::now();
    U_0_c = fft2D_pocket(convert_to_complex(U_r[0]));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "scatter_fast filter_bank time: " << diff.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vec2D_complex U_1_c = cdgmm_arma(U_0_c, filter_data.phi.levels[0]);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "cdgmm_arma time: " << diff.count() << std::endl;

    vec2D_complex subsample = subsample_fourier(U_1_c, std::pow(2, J));
    start = std::chrono::high_resolution_clock::now();
    vec2D_complex S_0 = ifft2D_pocket(subsample);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "ifft2D_pocket time: " << diff.count() << std::endl;

    vec2D_complex S_0_unpadded = unpad(S_0);

    std::vector<outputs> out_S_0;
    // std::vector<outputs> out_S_1;
    // std::vector<outputs> out_S_2;
    std::vector<outputs> out_S_1(filter_data.psi_filters.size(), {S_0_unpadded, {-1}, {-1}, {0}});
    std::vector<outputs> out_S_2(filter_data.psi_filters.size()*filter_data.psi_filters.size(), 
                                 {S_0_unpadded, {-1, -1}, {-1, -1}, {0, 0}});
    out_S_0.push_back({S_0_unpadded, {}, {}, {}});
    int max_order = 2;

    std::cout << "psi_filters size: " << filter_data.psi_filters.size() << std::endl;
    start = std::chrono::high_resolution_clock::now();
    /*
        calculate the scattering coefficients
    */
    //#pragma omp parallel for collapse(2) 
    #pragma omp parallel for 
    for (int n1 = 0; n1 < filter_data.psi_filters.size(); n1++)
    {
        int j1 = filter_data.psi_filters[n1].j;
        int theta1 = filter_data.psi_filters[n1].theta;

        vec2D_complex psi_f = filter_data.psi_filters[n1].levels[0];
        vec2D_complex phi_f = filter_data.phi.levels[j1];

        /*
            calculate the first order scattering coefficients
        */
        vec2D_complex U_1_c = cdgmm_arma(U_0_c, psi_f);
        if (j1 > 0)
            U_1_c = subsample_fourier(U_1_c, std::pow(2, j1));

        // inverse modulus and fft
        U_1_c = ifft2D_pocket(U_1_c);
        U_1_c = modulus(U_1_c);
        U_1_c = fft2D_pocket(U_1_c);

        // low pass filter
        vec2D_complex S_1_c = cdgmm_arma(U_1_c, phi_f);
        S_1_c = subsample_fourier(S_1_c, std::pow(2, (J - j1)));

        vec2D_complex S_1_r = ifft2D_pocket(S_1_c);
        S_1_r = unpad(S_1_r);

        // store the scattering coefficients
        //out_S_1.push_back({S_1_r, {j1}, {n1}, {theta1}});
        out_S_1[n1] = {S_1_r, {j1}, {n1}, {theta1}};

        if (max_order < 2)
            continue;

        /*
            calculate the second order scattering coefficients
        */
        //#pragma omp parallel for 
        for (int n2 = 0; n2 < filter_data.psi_filters.size(); n2++)
        {
            int j2 = filter_data.psi_filters[n2].j;
            int theta2 = filter_data.psi_filters[n2].theta;

            if (j2 <= j1)
                continue;

            vec2D_complex psi_f2 = filter_data.psi_filters[n2].levels[j1];
            vec2D_complex phi_f2 = filter_data.phi.levels[j2];

            vec2D_complex U_2_c = cdgmm_arma(U_1_c, psi_f2);
            U_2_c = subsample_fourier(U_2_c, std::pow(2, (j2 - j1)));

            // inverse modulus and fft
            U_2_c = ifft2D_pocket(U_2_c);
            U_2_c = modulus(U_2_c);
            U_2_c = fft2D_pocket(U_2_c);

            // low pass filter
            vec2D_complex S_2_c = cdgmm_arma(U_2_c, phi_f2);
            S_2_c = subsample_fourier(S_2_c, std::pow(2, (J - j2)));

            vec2D_complex S_2_r = ifft2D_pocket(S_2_c);
            S_2_r = unpad(S_2_r);

            // store the scattering coefficients
            //out_S_2.push_back({S_2_r, {j1, j2}, {n1, n2}, {theta1, theta2}});

            out_S_2[n2 + filter_data.psi_filters.size()*n1] = {S_2_r, {j1, j2}, {n1, n2}, {theta1, theta2}};
        }
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "scatter_fast loop time: " << diff.count() << std::endl;


    // join all the lists
    std::vector<outputs> out_S;
    out_S.insert(out_S.end(), out_S_0.begin(), out_S_0.end());

    for (auto it = out_S_1.begin(); it != out_S_1.end();)
    {
        if (it->n[0] < 0 ) 
        {
            out_S_1.erase(it);
            //std::cout << "out_S_1 erase item !!!" << std::endl;
        }
        else ++it;
    }
    out_S.insert(out_S.end(), out_S_1.begin(), out_S_1.end());

    for (auto it = out_S_2.begin(); it != out_S_2.end();)
    {
        if (it->n[0] < 0 && it->n[1] < 0) 
        {
            out_S_2.erase(it);
            //std::cout << "out_S_2 erase item !!!" << std::endl;
        }
        else ++it;
    }
    out_S.insert(out_S.end(), out_S_2.begin(), out_S_2.end());

    return out_S;
}

std::vector<outputs> scatter(vec2D signal, vec1D digit)
{
    /*
        specify the filter bank parameters
    */
    int M, N, J, L;
    M = N = 36;
    J = 2;
    L = 8;

    filterBank filter_data = filter_bank_fast(M, N, J, L);

    /*
        specify the input signal
    */
    vec3D U_r = pad(digit, 1);
    vec2D_complex U_0_c(36, vec1D_complex(36));
     
    auto start = std::chrono::high_resolution_clock::now();
    U_0_c = fft2D(convert_to_complex(U_r[0]));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "fft2D time: " << diff.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vec2D_complex U_1_c = cdgmm(U_0_c, filter_data.phi.levels[0]);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "cdgmm time: " << diff.count() << std::endl;

    vec2D_complex subsample = subsample_fourier(U_1_c, std::pow(2, J));
    start = std::chrono::high_resolution_clock::now();
    vec2D_complex S_0 = ifft2D(subsample);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "ifft2D time: " << diff.count() << std::endl;

    vec2D_complex S_0_unpadded = unpad(S_0);

    std::vector<outputs> out_S_0;
    std::vector<outputs> out_S_1;
    std::vector<outputs> out_S_2;

    out_S_0.push_back({S_0_unpadded, {}, {}, {}});
    int max_order = 2;

    /*
        calculate the scattering coefficients
    */
    for (int n1 = 0; n1 < filter_data.psi_filters.size(); n1++)
    {
        int j1 = filter_data.psi_filters[n1].j;
        int theta1 = filter_data.psi_filters[n1].theta;

        vec2D_complex psi_f = filter_data.psi_filters[n1].levels[0];
        vec2D_complex phi_f = filter_data.phi.levels[j1];

        /*
            calculate the first order scattering coefficients
        */
        vec2D_complex U_1_c = cdgmm(U_0_c, psi_f);
        if (j1 > 0)
            U_1_c = subsample_fourier(U_1_c, std::pow(2, j1));

        // inverse modulus and fft
        U_1_c = ifft2D(U_1_c);
        U_1_c = modulus(U_1_c);
        U_1_c = fft2D(U_1_c);

        // low pass filter
        vec2D_complex S_1_c = cdgmm(U_1_c, phi_f);
        S_1_c = subsample_fourier(S_1_c, std::pow(2, (J - j1)));

        vec2D_complex S_1_r = ifft2D(S_1_c);
        S_1_r = unpad(S_1_r);

        // store the scattering coefficients
        out_S_1.push_back({S_1_r, {j1}, {n1}, {theta1}});

        if (max_order < 2)
            continue;

        /*
            calculate the second order scattering coefficients
        */
        for (int n2 = 0; n2 < filter_data.psi_filters.size(); n2++)
        {
            int j2 = filter_data.psi_filters[n2].j;
            int theta2 = filter_data.psi_filters[n2].theta;

            if (j2 <= j1)
                continue;

            vec2D_complex psi_f2 = filter_data.psi_filters[n2].levels[j1];
            vec2D_complex phi_f2 = filter_data.phi.levels[j2];

            vec2D_complex U_2_c = cdgmm(U_1_c, psi_f2);
            U_2_c = subsample_fourier(U_2_c, std::pow(2, (j2 - j1)));

            // inverse modulus and fft
            U_2_c = ifft2D(U_2_c);
            U_2_c = modulus(U_2_c);
            U_2_c = fft2D(U_2_c);

            // low pass filter
            vec2D_complex S_2_c = cdgmm(U_2_c, phi_f2);
            S_2_c = subsample_fourier(S_2_c, std::pow(2, (J - j2)));

            vec2D_complex S_2_r = ifft2D(S_2_c);
            S_2_r = unpad(S_2_r);

            // store the scattering coefficients
            out_S_2.push_back({S_2_r, {j1, j2}, {n1, n2}, {theta1, theta2}});
        }
    }

    // join all the lists
    std::vector<outputs> out_S;
    out_S.insert(out_S.end(), out_S_0.begin(), out_S_0.end());
    out_S.insert(out_S.end(), out_S_1.begin(), out_S_1.end());
    out_S.insert(out_S.end(), out_S_2.begin(), out_S_2.end());

    return out_S;
}

std::vector<outputs> scatter(vec2D signal, vec1D digit, filterBank filter_data, int J)
{

    /*
        specify the input signal
    */
    vec3D U_r = pad(digit, 1);
    vec2D_complex U_0_c(36, vec1D_complex(36));
     
    auto start = std::chrono::high_resolution_clock::now();
    U_0_c = fft2D(convert_to_complex(U_r[0]));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "fft2D time: " << diff.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vec2D_complex U_1_c = cdgmm(U_0_c, filter_data.phi.levels[0]);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "cdgmm time: " << diff.count() << std::endl;

    vec2D_complex subsample = subsample_fourier(U_1_c, std::pow(2, J));
    start = std::chrono::high_resolution_clock::now();
    vec2D_complex S_0 = ifft2D(subsample);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "ifft2D time: " << diff.count() << std::endl;

    vec2D_complex S_0_unpadded = unpad(S_0);

    std::vector<outputs> out_S_0;
    std::vector<outputs> out_S_1;
    std::vector<outputs> out_S_2;

    out_S_0.push_back({S_0_unpadded, {}, {}, {}});
    int max_order = 2;

    /*
        calculate the scattering coefficients
    */
    for (int n1 = 0; n1 < filter_data.psi_filters.size(); n1++)
    {
        int j1 = filter_data.psi_filters[n1].j;
        int theta1 = filter_data.psi_filters[n1].theta;

        vec2D_complex psi_f = filter_data.psi_filters[n1].levels[0];
        vec2D_complex phi_f = filter_data.phi.levels[j1];

        /*
            calculate the first order scattering coefficients
        */
        vec2D_complex U_1_c = cdgmm(U_0_c, psi_f);
        if (j1 > 0)
            U_1_c = subsample_fourier(U_1_c, std::pow(2, j1));

        // inverse modulus and fft
        U_1_c = ifft2D(U_1_c);
        U_1_c = modulus(U_1_c);
        U_1_c = fft2D(U_1_c);

        // low pass filter
        vec2D_complex S_1_c = cdgmm(U_1_c, phi_f);
        S_1_c = subsample_fourier(S_1_c, std::pow(2, (J - j1)));

        vec2D_complex S_1_r = ifft2D(S_1_c);
        S_1_r = unpad(S_1_r);

        // store the scattering coefficients
        out_S_1.push_back({S_1_r, {j1}, {n1}, {theta1}});

        if (max_order < 2)
            continue;

        /*
            calculate the second order scattering coefficients
        */
        for (int n2 = 0; n2 < filter_data.psi_filters.size(); n2++)
        {
            int j2 = filter_data.psi_filters[n2].j;
            int theta2 = filter_data.psi_filters[n2].theta;

            if (j2 <= j1)
                continue;

            vec2D_complex psi_f2 = filter_data.psi_filters[n2].levels[j1];
            vec2D_complex phi_f2 = filter_data.phi.levels[j2];

            vec2D_complex U_2_c = cdgmm(U_1_c, psi_f2);
            U_2_c = subsample_fourier(U_2_c, std::pow(2, (j2 - j1)));

            // inverse modulus and fft
            U_2_c = ifft2D(U_2_c);
            U_2_c = modulus(U_2_c);
            U_2_c = fft2D(U_2_c);

            // low pass filter
            vec2D_complex S_2_c = cdgmm(U_2_c, phi_f2);
            S_2_c = subsample_fourier(S_2_c, std::pow(2, (J - j2)));

            vec2D_complex S_2_r = ifft2D(S_2_c);
            S_2_r = unpad(S_2_r);

            // store the scattering coefficients
            out_S_2.push_back({S_2_r, {j1, j2}, {n1, n2}, {theta1, theta2}});
        }
    }

    // join all the lists
    std::vector<outputs> out_S;
    out_S.insert(out_S.end(), out_S_0.begin(), out_S_0.end());
    out_S.insert(out_S.end(), out_S_1.begin(), out_S_1.end());
    out_S.insert(out_S.end(), out_S_2.begin(), out_S_2.end());

    return out_S;
}

// Softmax function
vec1D softmax(vec1D x)
{
    double max_x = *std::max_element(x.begin(), x.end());

    for (double &num : x)
    {
        num = std::exp(num - max_x);
    }

    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);

    for (double &num : x)
    {
        num /= sum_x;
    }

    return x;
}

// Prediction function
vec1D predict(const vec1D &x, const vec2D &coef, const vec1D &intercept)
{
    vec1D z(intercept.size());

    for (size_t i = 0; i < coef.size(); i++)
    {
        for (size_t j = 0; j < coef[i].size(); j++)
        {
            z[i] += coef[i][j] * x[j];
        }
        z[i] += intercept[i];
    }

    return softmax(z);
}

