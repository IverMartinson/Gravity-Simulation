#include <stdio.h>
#include <math.h>
#include <SDL2/SDL.h>

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 1000

__global__ void distance_kernal(float* d_positions, float* d_velocities, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length) return;

    float x = d_positions[i * 2];
    float y = d_positions[i * 2 + 1];
    float vx = d_velocities[i * 2];
    float vy = d_velocities[i * 2 + 1];

    float distance = sqrtf(x * x + y * y); 
    if (distance == 0) return;    

    float angle = atan2f(y, x);
    float force = 100.0f / distance;

    vx -= cosf(angle) * force;
    vy -= sinf(angle) * force;

    x += vx;
    y += vy;

    d_positions[i * 2] = x;
    d_positions[i * 2 + 1] = y;
    d_velocities[i * 2] = vx;
    d_velocities[i * 2 + 1] = vy;
}

float random(int Min, int Max)
{
    int diff = Max-Min;
    return (((float)(diff+1)/RAND_MAX) * rand() + Min);
}

int main(){
    int length = 200'000'000;

    float* positions = (float*)malloc(length * 2 * sizeof(float));
    float* velocities = (float*)malloc(length * 2 * sizeof(float));

    for (int i = 0; i < length; i++) {
        positions[i] = random(-500, -100);
        velocities[i * 2] = 1.0;
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Gravity Simulation (By Pixel Rendering, GPU Accelerated)",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_WIDTH,
                                          SCREEN_HEIGHT,
                                          SDL_WINDOW_SHOWN);
    if (window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    Uint32 start_time, frame_time;
    float fps;

    float* d_positions;
    float* d_velocities;

    cudaMalloc(&d_positions, length * 2 * sizeof(float));
    cudaMalloc(&d_velocities, length * 2 * sizeof(float));

    cudaMemcpy(d_positions, positions, length * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, length * 2 * sizeof(float), cudaMemcpyHostToDevice);
    bool running = true;
    SDL_Event event;
    while (running){
        start_time = SDL_GetTicks();
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
            }
        }


        dim3 threads(256);
        dim3 blocks((length + threads.x - 1) / threads.x);

        printf("block: %dx, %dy  thread: %dx, %dy\n", blocks.x, blocks.y, threads.x, threads.y);

        distance_kernal<<<blocks, threads>>>(d_positions, d_velocities, length);

        cudaMemcpy(positions, d_positions, length * sizeof(float), cudaMemcpyDeviceToHost);

        
        // Set the draw color to white
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

        // Clear the screen
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        SDL_RenderDrawPoint(renderer, 500, 500);
        SDL_RenderDrawPoint(renderer, 499, 500);
        SDL_RenderDrawPoint(renderer, 499, 499);
        SDL_RenderDrawPoint(renderer, 500, 499);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 10);

        for (int i = 0; i < length; i += 2) {
            int x = (int)positions[i];
            int y = (int)positions[i + 1];
            SDL_RenderDrawPoint(renderer, x + 500, y + 500);
        }

        // Update the screen
        SDL_RenderPresent(renderer);

        frame_time = SDL_GetTicks()-start_time;
        fps = (frame_time > 0) ? 1000.0f / frame_time : 0.0f;
        printf("%ffps\n", fps);
    }

    // Clean up
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    

    free(positions);
    cudaFree(d_positions);
    cudaFree(d_velocities);


    return 0;
}
