// Aadit Trivedi


#include <map>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstdlib>
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
};

// function to load stocks data from csv into backing data structure

std::vector<StockData> loadStocks(const std::string& filename) {
    std::vector<StockData> stocks;
    std::ifstream file(filename);
    std::string currentLine;

    // Skip the header
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


/*

Here, I start writing my aggregate queries

i.e. sums etc.

*/


// sum of the lowest prices
__global__ void sumLowPrice(StockData* stocks, float* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) { atomicAdd(result, stocks[index].lowPrice); }
}


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

    // experimenting with shared memory


    extern __shared__ StockData sharedData[];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    if (globalIdx < size) {
        sharedData[localIdx] = stocks[globalIdx];
    }
    // wait for threads to stop executing

    __syncthreads();
    if (globalIdx < size) {
        float max = sharedData[localIdx].highPrice;
        for (int i = 0; i < sizeOfWindow; i++) {
            if (sharedData[i + localIdx].highPrice > max) {
                max = sharedData[i + localIdx].highPrice;
            }
        }
        result[globalIdx] = max;
    }
}


int main() {

    // reading stocks.csv file
    
    std::vector<StockData> stocks = loadStocks("stocks.csv");
    StockData* d_stocks;
    bool* d_results;
    int size = stocks.size();
    float* agg_results;

    // moving to gpu


    cudaMalloc(&d_stocks, size * sizeof(StockData));
    cudaMalloc(&d_results, size * sizeof(bool));
    cudaMalloc(&agg_results, size * sizeof(float));
    cudaMemcpy(d_stocks, stocks.data(), size * sizeof(StockData), cudaMemcpyHostToDevice);

    auto printOutSpeedup = [](float cpuTime, float gpuTime) {
        std::cout << "Speedup: " << cpuTime / gpuTime << "x\n";
    };
    

    // EXPERIMENT 1.a:
    // CPU vs. GPU Filter by Volume

    std::cout << "Size: " << size << std::endl;
    std::cout << "---------------------\n";

    // cpu filter by volume
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<bool> results(size);
    for (int i = 0; i < size; i++) {
        results[i] = stocks[i].volume >= 1000000;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CPU filter by volume: " << elapsed.count() << "ms\n";

    // gpu filter by volume
    cudaEvent_t start_gpu, stop;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop);
    cudaEventRecord(start_gpu);
    filterStocksByVolume<<<(size + 255) / 256, 256>>>(d_stocks, d_results, size, 1000000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop);
    std::cout << "GPU filter by volume: " << milliseconds << "ms\n";
    printOutSpeedup(elapsed.count(), milliseconds);

    // accuracy check
    std::vector<char> resultsGpu(size);
    cudaMemcpy(resultsGpu.data(), d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (results[i] != resultsGpu[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match\n";
    }

    

    std::cout << "---------------------\n";



    // EXPERIMENT 1.b:
    // CPU vs. GPU Filter by Close Price

    // cpu filter by close price
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        results[i] = stocks[i].closePrice >= 100;
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start);
    std::cout << "CPU filter by close price: " << elapsed2.count() << "ms\n";

    // gpu filter by close price
    cudaEvent_t start_gpu2, stop2;
    cudaEventCreate(&start_gpu2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start_gpu2);
    filterStocksByClosePrice<<<(size + 255) / 256, 256>>>(d_stocks, d_results, size, 100);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu2, stop2);
    std::cout << "GPU filter by close price: " << milliseconds << "ms\n";
    printOutSpeedup(elapsed2.count(), milliseconds);

    std::vector<char> resultsGpu2(size);
    cudaMemcpy(resultsGpu2.data(), d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);
    match = true;
    for (int i = 0; i < size; i++) {
        if (results[i] != resultsGpu2[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match\n";
    }

    std::cout << "---------------------\n";

    
    // EXPERIMENT 1.c:
    // CPU vs. GPU Filter by Open Price

    // cpu filter by open price
    auto start22 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        results[i] = stocks[i].openPrice >= 100;
    }
    auto end22 = std::chrono::high_resolution_clock::now();
    auto elapsed22 = std::chrono::duration_cast<std::chrono::milliseconds>(end22 - start22);
    std::cout << "CPU filter by open price: " << elapsed22.count() << "ms\n";

    // gpu filter by open price
    cudaEvent_t start_gpu22, stop22;
    cudaEventCreate(&start_gpu22);
    cudaEventCreate(&stop22);
    cudaEventRecord(start_gpu22);
    filterStocksByOpenPrice<<<(size + 255) / 256, 256>>>(d_stocks, d_results, size, 100);
    cudaEventRecord(stop22);
    cudaEventSynchronize(stop22);
    float milliseconds22 = 0;
    cudaEventElapsedTime(&milliseconds22, start_gpu22, stop22);
    std::cout << "GPU filter by open price: " << milliseconds22 << "ms\n";
    printOutSpeedup(elapsed22.count(), milliseconds22);

    std::vector<char> resultsGpu22(size);
    cudaMemcpy(resultsGpu22.data(), d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);
    match = true;
    for (int i = 0; i < size; i++) {
        if (results[i] != resultsGpu22[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match\n";
    }

    std::cout << "---------------------\n";

    // EXPERIMENT 1.d:
    // CPU vs. GPU Filter by High Price

    // cpu filter by high price
    auto start3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        results[i] = stocks[i].highPrice >= 100;
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto elapsed3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "CPU filter by high price: " << elapsed3.count() << "ms\n";

    // gpu filter by high price
    cudaEvent_t start_gpu3, stop3;
    cudaEventCreate(&start_gpu3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start_gpu3);
    filterStocksByHighPrice<<<(size + 255) / 256, 256>>>(d_stocks, d_results, size, 100);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start_gpu3, stop3);
    std::cout << "GPU filter by high price: " << milliseconds3 << "ms\n";
    printOutSpeedup(elapsed3.count(), milliseconds3);

    std::vector<char> resultsGpu3(size);
    cudaMemcpy(resultsGpu3.data(), d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);
    match = true;
    for (int i = 0; i < size; i++) {
        if (results[i] != resultsGpu3[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match\n";
    }

    std::cout << "---------------------\n";

    // EXPERIMENT 1.e:
    // CPU vs. GPU Filter by Low Price

    // cpu filter by low price
    auto start4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        results[i] = stocks[i].lowPrice >= 100;
    }
    auto end4 = std::chrono::high_resolution_clock::now();
    auto elapsed4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4);
    std::cout << "CPU filter by low price: " << elapsed4.count() << "ms\n";

    // gpu filter by low price
    cudaEvent_t start_gpu4, stop4;
    cudaEventCreate(&start_gpu4);
    cudaEventCreate(&stop4);
    cudaEventRecord(start_gpu4);
    filterStocksByLowPrice<<<(size + 255) / 256, 256>>>(d_stocks, d_results, size, 100);
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);
    float milliseconds4 = 0;
    cudaEventElapsedTime(&milliseconds4, start_gpu4, stop4);
    std::cout << "GPU filter by low price: " << milliseconds4 << "ms\n";
    printOutSpeedup(elapsed4.count(), milliseconds4);

    std::vector<char> resultsGpu4(size);
    cudaMemcpy(resultsGpu4.data(), d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);
    match = true;
    for (int i = 0; i < size; i++) {
        if (results[i] != resultsGpu4[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match\n";
    }

    std::cout << "---------------------\n";

    // EXPERIMENT 1.f:
    // CPU vs. GPU Filter by High Price and Low Price

    // cpu filter by high price and low price
    auto start5 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        results[i] = stocks[i].highPrice >= 100 && stocks[i].lowPrice >= 100;
    }
    auto end5 = std::chrono::high_resolution_clock::now();
    auto elapsed5 = std::chrono::duration_cast<std::chrono::milliseconds>(end5 - start5);
    std::cout << "CPU filter by high price and low price: " << elapsed5.count() << "ms\n";

    // gpu filter by high price and low price
    cudaEvent_t start_gpu5, stop5;
    cudaEventCreate(&start_gpu5);
    cudaEventCreate(&stop5);
    cudaEventRecord(start_gpu5);
    filterStocksByHighPriceAndLowPrice<<<(size + 255) / 256, 256>>>(d_stocks, d_results, size, 100, 100);
    cudaEventRecord(stop5);
    cudaEventSynchronize(stop5);
    float milliseconds5 = 0;
    cudaEventElapsedTime(&milliseconds5, start_gpu5, stop5);
    std::cout << "GPU filter by high price and low price: " << milliseconds5 << "ms\n";
    printOutSpeedup(elapsed5.count(), milliseconds5);

    std::vector<char> resultsGpu5(size);
    cudaMemcpy(resultsGpu5.data(), d_results, size * sizeof(bool), cudaMemcpyDeviceToHost);
    match = true;
    for (int i = 0; i < size; i++) {
        if (results[i] != resultsGpu5[i]) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match\n";
    }

    std::cout << "---------------------\n";
    std::cout << "\n";

    // EXPERIMENT 2.a:
    // CPU vs. GPU Sum of Low Prices

    // cpu sum of low prices
    auto start6 = std::chrono::high_resolution_clock::now();
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += stocks[i].lowPrice;
    }
    auto end6 = std::chrono::high_resolution_clock::now();
    auto elapsed6 = std::chrono::duration_cast<std::chrono::milliseconds>(end6 - start6);
    std::cout << "CPU sum of low prices: " << elapsed6.count() << "ms\n";

    // gpu sum of low prices
    cudaEvent_t start_gpu6, stop6;
    cudaEventCreate(&start_gpu6);
    cudaEventCreate(&stop6);
    cudaEventRecord(start_gpu6);
    sumLowPrice<<<(size + 255) / 256, 256>>>(d_stocks, agg_results, size);
    cudaEventRecord(stop6);
    cudaEventSynchronize(stop6);
    float milliseconds6 = 0;
    cudaEventElapsedTime(&milliseconds6, start_gpu6, stop6);
    std::cout << "GPU sum of low prices: " << milliseconds6 << "ms\n";
    std::cout << "Speedup: " << elapsed6.count() / milliseconds6 << "x\n";

    std::vector<float> resultsGpu6(size);
    cudaMemcpy(resultsGpu6.data(), agg_results, size * sizeof(float), cudaMemcpyDeviceToHost);
    float sumGpu = 0;
    for (int i = 0; i < size; i++) {
        sumGpu += resultsGpu6[i];
    }
    if (sum == sumGpu) {
        std::cout << "Results match\n";
    }


    std::cout << "---------------------\n";

    // EXPERIMENT 2.b:
    // CPU vs. GPU Moving Average

    // cpu moving average
    auto start7 = std::chrono::high_resolution_clock::now();
    std::vector<float> movingAverage(size);
    for (int i = 0; i < size - 10; i++) { // Ensure no out-of-bounds access
        float sum = 0;
        for (int j = 0; j < 10; j++) {
            sum += stocks[i + j].closePrice;
        }
        movingAverage[i] = sum / 10;
    }
    auto end7 = std::chrono::high_resolution_clock::now();
    auto elapsed7 = std::chrono::duration_cast<std::chrono::milliseconds>(end7 - start7);
    std::cout << "CPU moving average: " << elapsed7.count() << "ms\n";

    // gpu moving average
    cudaEvent_t start_gpu7, stop7;
    cudaEventCreate(&start_gpu7);
    cudaEventCreate(&stop7);
    cudaEventRecord(start_gpu7);
    getMovingAverage<<<(size + 255) / 256, 256>>>(d_stocks, agg_results, size, 10);
    cudaEventRecord(stop7);
    cudaEventSynchronize(stop7);
    float milliseconds7 = 0;
    cudaEventElapsedTime(&milliseconds7, start_gpu7, stop7);
    std::cout << "GPU moving average: " << milliseconds7 << "ms\n";
    std::cout << "Speedup: " << elapsed7.count() / milliseconds7 << "x\n";

    std::vector<float> resultsGpu7(size);
    cudaMemcpy(resultsGpu7.data(), agg_results, size * sizeof(float), cudaMemcpyDeviceToHost);
    float sumGpu7 = 0;
    for (int i = 0; i < size; i++) {
        sumGpu7 += resultsGpu7[i];
    }
    if (sum == sumGpu7) {
        std::cout << "Results match\n";
    }


    std::cout << "---------------------\n";

    // EXPERIMENT 2.c:
    // CPU vs. GPU Minimum Low Price

    // cpu minimum low price
    auto start8 = std::chrono::high_resolution_clock::now();
    float minLowPrice = stocks[0].lowPrice;
    for (int i = 0; i < size; i++) {
        if (stocks[i].lowPrice < minLowPrice) {
            minLowPrice = stocks[i].lowPrice;
        }
    }
    auto end8 = std::chrono::high_resolution_clock::now();
    auto elapsed8 = std::chrono::duration_cast<std::chrono::milliseconds>(end8 - start8);
    std::cout << "CPU minimum low price: " << elapsed8.count() << "ms\n";

    // gpu minimum low price
    cudaEvent_t start_gpu8, stop8;
    cudaEventCreate(&start_gpu8);
    cudaEventCreate(&stop8);
    cudaEventRecord(start_gpu8);
    getMinLowPrice<<<(size + 255) / 256, 256>>>(d_stocks, agg_results, size, 10);
    cudaEventRecord(stop8);
    cudaEventSynchronize(stop8);
    float milliseconds8 = 0;
    cudaEventElapsedTime(&milliseconds8, start_gpu8, stop8);
    std::cout << "GPU minimum low price: " << milliseconds8 << "ms\n";
    std::cout << "Speedup: " << elapsed8.count() / milliseconds8 << "x\n";

    std::vector<float> resultsGpu8(size);
    cudaMemcpy(resultsGpu8.data(), agg_results, size * sizeof(float), cudaMemcpyDeviceToHost);
    float minGpu = resultsGpu8[0];
    for (int i = 0; i < size; i++) {
        if (resultsGpu8[i] < minGpu) {
            minGpu = resultsGpu8[i];
        }
    }
    if (minLowPrice == minGpu) {
        std::cout << "Results match\n";
    }


    std::cout << "---------------------\n";

    // EXPERIMENT 2.d:
    // CPU vs. GPU Maximum High Price

    // cpu maximum high price
    auto start9 = std::chrono::high_resolution_clock::now();
    float maxHighPrice = stocks[0].highPrice;
    for (int i = 0; i < size; i++) {
        if (stocks[i].highPrice > maxHighPrice) {
            maxHighPrice = stocks[i].highPrice;
        }
    }
    auto end9 = std::chrono::high_resolution_clock::now();
    auto elapsed9 = std::chrono::duration_cast<std::chrono::milliseconds>(end9 - start9);
    std::cout << "CPU maximum high price: " << elapsed9.count() << "ms\n";

    // gpu maximum high price
    cudaEvent_t start_gpu9, stop9;
    cudaEventCreate(&start_gpu9);
    cudaEventCreate(&stop9);
    cudaEventRecord(start_gpu9);
    getMaxHighPrice<<<(size + 255) / 256, 256, 256 * sizeof(StockData)>>>(d_stocks, agg_results, size, 10);
    cudaEventRecord(stop9);
    cudaEventSynchronize(stop9);
    float milliseconds9 = 0;
    cudaEventElapsedTime(&milliseconds9, start_gpu9, stop9);
    std::cout << "GPU maximum high price: " << milliseconds9 << "ms\n";
    std::cout << "Speedup: " << elapsed9.count() / milliseconds9 << "x\n";

    std::vector<float> resultsGpu9(size);
    cudaMemcpy(resultsGpu9.data(), agg_results, size * sizeof(float), cudaMemcpyDeviceToHost);
    float maxGpu = resultsGpu9[0];
    for (int i = 0; i < size; i++) {
        if (resultsGpu9[i] > maxGpu) {
            maxGpu = resultsGpu9[i];
        }
    }
    if (maxHighPrice == maxGpu) {
        std::cout << "Results match\n";
    }
    std::cout << "---------------------\n";
    std::cout << "\n";


    std::cout << "----------sorted vector-based index-----------\n";


    // EXPERIMENT 3

    /*
    here i try using a sorted vector list as an index
    */

    std::vector<StockData> stocks2(10000000);
    for (int i = 0; i < stocks2.size(); ++i) {
        stocks2[i] = {
            "2021-01-01",
            100.0f + rand() % 100,
            1000000.0f + rand() % 1000000,
            100.0f + rand() % 100,
            100.0f + rand() % 100,
            100.0f + rand() % 100
        };
    }
    auto startIndexBuild = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<float, StockData>> sortedIndex;
    for (const auto& stock : stocks2) {
        sortedIndex.emplace_back(stock.volume, stock);
    }
    std::sort(sortedIndex.begin(), sortedIndex.end(),
              [](const std::pair<float, StockData>& a, const std::pair<float, StockData>& b) {
                  return a.first < b.first;
              });
    auto endIndexBuild = std::chrono::high_resolution_clock::now();
    auto elapsedIndexBuild = std::chrono::duration_cast<std::chrono::milliseconds>(endIndexBuild - startIndexBuild);
    std::cout << "Index built in: " << elapsedIndexBuild.count() << "ms\n";

    // Filter with index
    auto startFilteringWithIndex2 = std::chrono::high_resolution_clock::now();
    std::vector<StockData> filteredStocks2;
    auto lowerBound = std::lower_bound(sortedIndex.begin(), sortedIndex.end(), 1800000.0f,
        [](const std::pair<float, StockData>& pair, float value) {
            return pair.first < value;
        }
    );
    for (auto it = lowerBound; it != sortedIndex.end(); ++it) {
        filteredStocks2.push_back(it->second);
    }
    auto endFilteringWithIndex2 = std::chrono::high_resolution_clock::now();
    auto elapsedFilteringWithIndex2 = std::chrono::duration_cast<std::chrono::milliseconds>(endFilteringWithIndex2 - startFilteringWithIndex2);
    std::cout << "Filtering with index: " << elapsedFilteringWithIndex2.count() << "ms\n";

    // Linear search
    auto startLinearSearch = std::chrono::high_resolution_clock::now();
    std::vector<StockData> filteredStocksLinearSearch;
    for (const StockData& stock : stocks2) {
        if (stock.volume >= 1800000) {
            filteredStocksLinearSearch.push_back(stock);
        }
    }
    auto endLinearSearch = std::chrono::high_resolution_clock::now();
    auto elapsedLinearSearch = std::chrono::duration_cast<std::chrono::milliseconds>(endLinearSearch - startLinearSearch);
    std::cout << "Linear search: " << elapsedLinearSearch.count() << "ms\n";





    // EXPERIMENT 3.1

    /*
    here i implmement a map-based index
    std::map used red black tree as backing structure
    */

    std::cout << "----------map-based index-----------\n";
    auto startIndexBuildMap = std::chrono::high_resolution_clock::now();
    std::map<float, StockData> sortedIndexMap;
    for (const auto& stock : stocks2) {
        sortedIndexMap[stock.volume] = stock;
    }
    auto endIndexBuildMap = std::chrono::high_resolution_clock::now();
    auto elapsedIndexBuildMap = std::chrono::duration_cast<std::chrono::milliseconds>(endIndexBuildMap - startIndexBuildMap);
    std::cout << "Index built in: " << elapsedIndexBuildMap.count() << "ms\n";

    // map-based filter here
    auto filterWithMapIndex = std::chrono::high_resolution_clock::now();
    std::vector<StockData> filteredStocksMap;
    for (auto it = sortedIndexMap.lower_bound(1800000.0f); it != sortedIndexMap.end(); ++it) {
        filteredStocksMap.push_back(it->second);
    }
    auto endFilteringWithMapIndex = std::chrono::high_resolution_clock::now();
    auto elapsedFilteringWithMapIndex = std::chrono::duration_cast<std::chrono::milliseconds>(endFilteringWithMapIndex - filterWithMapIndex);
    std::cout << "Filtering with map index: " << elapsedFilteringWithMapIndex.count() << "ms\n";

    // linear search
    auto startLinearSearchMap = std::chrono::high_resolution_clock::now();
    std::vector<StockData> filteredStocksLinearSearchMap;
    for (const StockData& stock : stocks2) {
        if (stock.volume >= 1800000) {
            filteredStocksLinearSearchMap.push_back(stock);
        }
    }
    auto endLinearSearchMap = std::chrono::high_resolution_clock::now();
    auto elapsedLinearSearchMap = std::chrono::duration_cast<std::chrono::milliseconds>(endLinearSearchMap - startLinearSearchMap);
    std::cout << "Linear search: " << elapsedLinearSearchMap.count() << "ms\n";
    std::cout << "\n";

    return 0;

}