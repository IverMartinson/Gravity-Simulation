#include <stdio.h>
#include <math.h>
#include <SDL2/SDL.h>

#define WIDTH 1000
#define HEIGHT 1000

__global__ void distance_kernal(float* d_positions, float* d_velocities, Uint32* d_pixels, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length) return;

    float x = d_positions[i * 2];
    float y = d_positions[i * 2 + 1];
    float vx = d_velocities[i * 2];
    float vy = d_velocities[i * 2 + 1];

    float distance = sqrtf(x * x + y * y); 
    if (distance == 0) return;    

    float angle = atan2f(y, x);
    float force = 1000.0f / distance;

    vx -= cosf(angle) * force;
    vy -= sinf(angle) * force;

    x += vx;
    y += vy;

    d_positions[i * 2] = x;
    d_positions[i * 2 + 1] = y;
    d_velocities[i * 2] = vx;
    d_velocities[i * 2 + 1] = vy;

    int x_ = (int)x / 10 + 500;
    int y_ = (int)y / 10 + 500;
    
    if (x_ < 0 || x_ >= WIDTH || y_ < 0 || y_ >= HEIGHT) {
        return;
    }

    Uint32& destColor = d_pixels[y_ * WIDTH + x_];

    int srcColor = 255;
    int srcAlpha = 5;

    Uint8 destR = (destColor >> 16) & 0xFF;

    Uint8 newR = ((srcColor - destR) * srcAlpha) / 255 + destR;

    destColor = (newR * 0x01010101);

}

float random(int Min, int Max)
{
    int diff = Max-Min;
    return (((float)(diff+1)/RAND_MAX) * rand() + Min);
}

SDL_Window* create_window(const char* title, int width, int height){
    SDL_Window* window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    
    return window;
}

int main(){
    int length = 250'000'000;

    float* positions = (float*)malloc(length * 2 * sizeof(float));
    float* velocities = (float*)malloc(length * 2 * sizeof(float));

    for (int i = 0; i < length; i++) {
        positions[i * 2] = random(-100000, 0);
        positions[i * 2 + 1] = random(-5000, 5000);
        velocities[i * 2] = 10;
        velocities[i * 2 + 1] = 0;
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = create_window("Gravity Simulation (Texture Rendering, GPU Accelerated)", WIDTH, HEIGHT);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    Uint32* pixels = new Uint32[WIDTH * HEIGHT];

    Uint32 start_time, frame_time;
    float fps;

    float* d_positions;
    float* d_velocities;
    Uint32* d_pixels;

    cudaMalloc(&d_positions, length * 2 * sizeof(float));
    cudaMalloc(&d_velocities, length * 2 * sizeof(float));

    cudaMalloc(&d_pixels, WIDTH * HEIGHT * sizeof(Uint32));
    cudaMemcpy(d_pixels, pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyHostToDevice);


    cudaMemcpy(d_positions, positions, length * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, length * 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);


    float max_fps = 0;

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


        dim3 threads(64);
        dim3 blocks((length + threads.x - 1) / threads.x);

        printf("block: %dx, %dy  thread: %dx, %dy\n", blocks.x, blocks.y, threads.x, threads.y);
        
        memset(pixels, 0, WIDTH * HEIGHT * sizeof(Uint32));
        cudaMemcpy(d_pixels, pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyHostToDevice);

        distance_kernal<<<blocks, threads>>>(d_positions, d_velocities, d_pixels, length);

        cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);

        pixels[500500] = 0xFFFFFFFF;

        void* mPixels;
        int pitch;

        SDL_LockTexture(texture, NULL, &mPixels, &pitch);
        memcpy(mPixels, pixels, WIDTH * HEIGHT * sizeof(Uint32));
        SDL_UnlockTexture(texture);

        SDL_RenderCopy(renderer, texture, NULL, NULL);

        SDL_RenderPresent(renderer);

        frame_time = SDL_GetTicks()-start_time;
        fps = (frame_time > 0) ? 1000.0f / frame_time : 0.0f;
        if (fps > max_fps){
            max_fps = fps;
        }
        printf("%d fps \n%d max fps\n", (int)fps, (int)max_fps);
        printf("%d particles\n", length);
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
