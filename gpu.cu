#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>



#define NUM_THREADS 256

thrust::device_ptr<int> particles;
thrust::device_ptr<int> offset;
thrust::device_ptr<int> bin_count;

int bins_size;
int grid;
int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}






__global__ void compute_forces_gpu(particle_t* parts, int num_parts, int* particles, int* offset, int grid) {
    // Get bin  ID
    int bin_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bin_idx >= grid * grid)
        return;

    const int row = bin_idx / grid;
    const int col = bin_idx % grid;
    int start = offset[bin_idx];
    int end = offset[bin_idx + 1];

    // Iterating neighboring bins
    for (int neighbor_row = row -1; neighbor_row <=row + 1; neighbor_row++) {
        for (int neighbor_col =col-1; neighbor_col <= col+1; neighbor_col++) {


            
            if (neighbor_row >= 0 && neighbor_row < grid && neighbor_col >= 0 && neighbor_col < grid) {
                const int neighbor_bin_idx = neighbor_row * grid + neighbor_col;

                // Iterate over particles in the current bin and its neighbor bin
                for (int i = start; i <end; i++) {
                    for (int j = offset[neighbor_bin_idx]; j < offset[neighbor_bin_idx + 1]; j++) {
                        // Apply force between particle i in the current bin and particle j in the neighbor bin
                        apply_force_gpu(parts[particles[i]], parts[particles[j]]);
                    }
                }
            }
        }
    }
}


__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}
__global__ void fill_count(int* bin_count, int num_parts, double size, particle_t* parts, int grid) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
        }
    int row =floor( parts[tid].x / cutoff);
    int col = floor(parts[tid].y / cutoff);
    parts[tid].ax = parts[tid].ay = 0;
    int bin_id = row * grid + col;
    atomicAdd(&bin_count[bin_id], 1);
    
}

__global__ void binning(particle_t* parts, int num_parts, int* particles, int* offset, int grid) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int x = floor(parts[tid].x / cutoff);
    int y = floor(parts[tid].y / cutoff);
    int bin = x * grid + y;

    int index = atomicAdd(&offset[bin], 1);
    //Offset gets changed 

    particles[index] = tid;
}


void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    grid = ceil(size / cutoff);
    bins_size = grid * grid;
    bin_count = thrust::device_malloc<int>(bins_size);
    particles = thrust::device_malloc<int>(num_parts);
    offset = thrust::device_malloc<int>(bins_size + 1);
}

void init_step() {
    thrust::fill(bin_count, bin_count + bins_size, 0);
}

void calc_cumulative_offset_step() {
    thrust::inclusive_scan(bin_count, bin_count + bins_size, bin_count);
}


void simulate_one_step(particle_t* parts, int num_parts, double size) {

    //Clear the bin_count
    init_step();
    // Count particles per bin
    fill_count<<<blks, NUM_THREADS>>>(bin_count.get(), num_parts, size, parts, grid);
    cudaDeviceSynchronize();

    // Perform inclusive scan
    calc_cumulative_offset_step();

    //Initialize offset with 0
    cudaMemset(offset.get(), 0, bins_size+1);
    //Copy prefix sums to offset starting from offset[1]
    cudaMemcpy(offset.get() + 1, bin_count.get(), bins_size * sizeof(int), cudaMemcpyDeviceToDevice);

    //Fill the particles according to the bins
    binning<<<blks, NUM_THREADS>>>(parts, num_parts, particles.get(), offset.get(), grid);

    //Reset Offsets
    cudaMemset(offset.get(), 0, bins_size+1);
    cudaMemcpy(offset.get() + 1, bin_count.get(), bins_size * sizeof(int), cudaMemcpyDeviceToDevice);


    // Call kernel to compute forces and move particles
    compute_forces_gpu<<<(grid * grid + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(parts, num_parts, particles.get(), offset.get(), grid);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}

// Clear allocations
void clear_simulation() {
    thrust::device_free(bin_count);
    thrust::device_free (particles);
    thrust::device_free (offset);
}