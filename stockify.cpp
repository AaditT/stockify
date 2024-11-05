// Aadit Trivedi

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


#include <cuda.h>
#include <cuda_runtime.h>


// stock struct
// backing data structure
// has date, close price, volume, open price, high price, low price

struct StockData {
    std::string date;
    float closePrice;
    float volume;
    float openPrice;
    float highPrice;
    float lowPrice;
}

// function to load stocks data from csv into backing data structure

std::vector<StockData> loadStocks(const std::string& filename) {

    std::vector <StockData> stocks;
    std::ifstream file(filename);
    std::string currentLine;
    std::getline(file, currentLine);

    while (std::getline(file, currentLine)) {

        std::stringstream ss(currentLine);
        StockData stock;

        std::getline(ss, stock.date, ',');
        std::string closePrice;
        std::getline(ss, closePrice, ',');
        stock.closePrice = std::stof(closePrice);
        std::string volume;
        std::getline(ss, volume, ',');
        stock.volume = std::stof(volume);
        std::string openPrice;
        std::getline(ss, openPrice, ',');
        stock.openPrice = std::stof(openPrice);
        std::string highPrice;
        std::getline(ss, highPrice, ',');
        stock.highPrice = std::stof(highPrice);
        std::string lowPrice;
        std::getline(ss, lowPrice, ',');
        stock.lowPrice = std::stof(lowPrice);
        
        stocks.push_back(stock);
    }

    file.close();
    return stocks;
}


/*

Here I define all my simpler query functions

SELECT * FROM stocks WHERE filter1, filter2, filter3

filter1, filter2, filter3 basically represent equality comparisons here
=, >, >=, <, <=, != etc.

*/

// filter functions
__global__ void filterStocksByVolume(StockData* stocks, bool* results, int size, float minVolume) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { results[index] = stocks[index].volume >= minVolume; }
}


__global__ void filterStocksByClosePrice(StockData* stocks, bool* results, int size, float minClosePrice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { results[index] = stocks[index].closePrice >= minClosePrice; }
}


__global__ void filterStocksByOpenPrice(StockData* stocks, bool* results, int size, float minOpenPrice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { results[index] = stocks[index].openPrice >= minOpenPrice; }
}


__global__ void filterStocksByHighPrice(StockData* stocks, bool* results, int size, float minHighPrice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { results[index] = stocks[index].highPrice >= minHighPrice; }
}


__global__ void filterStocksByLowPrice(StockData* stocks, bool* results, int size, float minLowPrice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { results[index] = stocks[index].lowPrice >= minLowPrice; }
}


__global__ void filterStocksByHighPriceAndLowPrice(StockData* stocks, bool* results, int size, float minHighPrice, float minLowPrice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { results[index] = stocks[index].highPrice >= minHighPrice && stocks[index].lowPrice >= minLowPrice; }
}


// sum of the lowest prices
__global__ void sumLowPrice(StockData* stocks, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { atomicAdd(result, stocks[index].lowPrice); }
}



/*

Here, I start writing my aggregate queries

i.e. sums etc.

*/


// get the moving avg
__global__ void getMovingAverage(StockData* stocks, float* result, int size, int sizeOfWindow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = 0;
        for (int i = 0; i < sizeOfWindow; i++) { sum += stocks[i + i].closePrice; }
        result[i] = sum / sizeOfWindow;
    }
}


// get minimum low price
__global__ void getMinLowPrice(StockData* stocks, float* result, int size, int sizeOfWindow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float min = stocks[i].lowPrice;
        for (int i = 0; i < sizeOfWindow; i++) {
            if (stocks[i + i].lowPrice < min) { min = stocks[i + i].lowPrice; }
        }
        result[i] = min;
    }
}


// get maximum high price
__global__ void getMaxHighPrice(StockData* stocks, float* result, int size, int sizeOfWindow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float max = stocks[i].highPrice;
        for (int i = 0; i < sizeOfWindow; i++) {
            if (stocks[i + i].highPrice > max) { max = stocks[i + i].highPrice; }
        }
        result[i] = max;
    }
}


int main() {

    // reading stocks.csv file
    
    std::vector<StockData> stocks = loadStocks("stock_data.csv");
    StockData* d_stocks;
    bool* d_results;

    // move both stocks and results to the device
    cudaMalloc(&d_stocks, stocks.size() * sizeof(StockData));
    cudaMalloc(&d_results, stocks.size() * sizeof(bool));
    cudaMemcpy(d_stocks, stocks.data(), stocks.size() * sizeof(StockData), cudaMemcpyHostToDevice);

    // Testing goes here..

    /*
    
    
    test diff queries

    evaulate performance comparison

    try gpu optimization here? along with multi threading
    
    etc.
    
    
    */


    cudaFree(d_stocks);
    cudaFree(d_results);
    return 0;


}