// N body simulation using CUDA
// each body has position, velocity, mass associated with a row in a matrix
// each thread computes the forces on one body due to all other bodies

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float3 position;
    float3 velocity;
    float mass;
} Body;

__global__ void update_bodies(Body* bodies, int nums_body, float time_step) {
    const int epsilon = 1e-10f;
    const int grav_const = 1; // gravitational constant

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nums_body) {
        float3 force = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < nums_body; i++) {
            if (i != idx) {
                float3 dir = make_float3(bodies[i].position.x - bodies[idx].position.x,
                                         bodies[i].position.y - bodies[idx].position.y,
                                         bodies[i].position.z - bodies[idx].position.z);
                float dist_sqr = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + epsilon; // avoid div by zero
                float inv_dist = rsqrtf(dist_sqr);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;  // instead of normalizing dir (has magnitude same as dist) 
                                                                   // and then dividing by dist^2,
                                                                   // we can multiply by 1/dist^3 to get the same result
                float f = bodies[i].mass * inv_dist3 * grav_const; // gravitational constant G is assumed to be 1
                force.x += f * dir.x; 
                force.y += f * dir.y;
                force.z += f * dir.z;
            }
        }
        // Update velocity and position
        bodies[idx].velocity.x += force.x / bodies[idx].mass * time_step;
        bodies[idx].velocity.y += force.y / bodies[idx].mass * time_step;
        bodies[idx].velocity.z += force.z / bodies[idx].mass * time_step;
        bodies[idx].position.x += bodies[idx].velocity.x * time_step;
        bodies[idx].position.y += bodies[idx].velocity.y * time_step;
        bodies[idx].position.z += bodies[idx].velocity.z * time_step;
    }
}

int main() {
    int nums_body = 5;
    Body bodies[nums_body];
    srand(0);
    for (int i = 0; i < nums_body; i++) {
        bodies[i].position = make_float3(static_cast<float>(rand()),
                                         static_cast<float>(rand()),
                                         static_cast<float>(rand()));
        bodies[i].velocity = make_float3(static_cast<float>(rand()),
                                         static_cast<float>(rand()),
                                         static_cast<float>(rand()));
        bodies[i].mass = static_cast<float>(rand());
    }

    size_t body_type_size = sizeof(Body);

    // allocate device memory, copy over bodies[] to do calculations on GPU
    // each thread computes the forces on one body due to all other bodies
    Body* d_bodies;
    cudaMalloc((void**)&d_bodies, nums_body * body_type_size);
    cudaMemcpy(d_bodies, bodies, nums_body * body_type_size, cudaMemcpyHostToDevice);

    for (int step = 0; step < 100; step++) {
        update_bodies<<<1, nums_body>>>(d_bodies, nums_body, 0.01f);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(bodies, d_bodies, nums_body * body_type_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_bodies);

    return EXIT_SUCCESS;
}

