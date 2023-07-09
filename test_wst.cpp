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

#include "wst.h"

/*

./test_wst ../data/t10k-images-idx3-ubyte  ../data/model.txt -1

*/

int main(int argc, char* argv[])
{
    
    vec2D signal;
    vec1D digit;

    int char_number = atoi(argv[3]);
    std::tie(signal, digit) = load(argv[1], char_number);

    /*
        specify the filter bank parameters
    */
    int M, N, J, L;
    J = 2;
    M = N = compute_padding(28, J);
    
    L = 8;

    filterBank filter_data = filter_bank(M, N, J, L);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<outputs> scattering_output = scatter(signal, digit, filter_data, J);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "scatter time: " << diff.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    scattering_output = scatter_fast(signal, digit, filter_data, J);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "scatter_fast time: " << diff.count() << std::endl;

    vec2D coefficients;
    vec1D intercept;

    std::ifstream fin(argv[2]);
    if (!fin)
    {
        std::cerr << "failed to open model file" << std::endl;
        return -1;
    }

    std::string line;

    // Read coefficients
    if (std::getline(fin, line))
    {
        // Remove the brackets at both ends
        line = line.substr(1, line.length() - 2);

        std::stringstream ss(line);
        std::string item;
        std::string row;

        vec1D temp;

        // Loop over each comma-separated value
        while (std::getline(ss, item, ','))
        {
            // trim any whitespace
            item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());

            // Check for start and end of row
            if (item.front() == '[')
            {
                // A new row begins, clear temp vector for new values
                temp.clear();
                item = item.substr(1);
            }
            if (item.back() == ']')
            {
                // A row ends, add the row to coefficients and continue to next item
                item.pop_back();
                temp.push_back(std::stod(item));
                coefficients.push_back(temp);
                continue;
            }

            // Convert string to double and add to vector
            temp.push_back(std::stod(item));
        }
    }

    // Read intercepts
    if (std::getline(fin, line))
    {
        // Remove the brackets at both ends
        line = line.substr(1, line.length() - 2);

        std::stringstream ss(line);
        std::string item;

        // Loop over each comma-separated value
        while (std::getline(ss, item, ','))
        {
            // trim any whitespace
            item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());

            // Convert string to double and add to vector
            intercept.push_back(std::stod(item));
        }
    }

    // close the file
    fin.close();

    start = std::chrono::high_resolution_clock::now();
    // flatten scattering_output into a vector of coefficients
    vec1D x;
    for (auto &out : scattering_output)
    {
        for (auto &s : out.coef)
        {
            for (auto &row : s)
            {
                x.push_back(row.real());
            }
        }
    }

    
    // Make the prediction
    std::vector<double> y_pred = predict(x, coefficients, intercept);

    // get index of max element
    auto max_it = std::max_element(y_pred.begin(), y_pred.end());
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "predict time: " << diff.count() << std::endl;

    // Print the output like [index]: [value]
    for (size_t i = 0; i < y_pred.size(); i++)
    {

        if (y_pred[i] == *max_it)
        {
            std::cout << i << ": " << std::fixed << std::setprecision(2) << y_pred[i] << " << 🙂" << std::endl;
            continue;
        }
        std::cout << i << ": " << std::fixed << std::setprecision(2) << y_pred[i] << std::endl;
    }
}