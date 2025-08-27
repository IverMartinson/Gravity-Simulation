#include <stdio.h>
#include <SDL2/SDL.h>
#include <math.h>
#include <time.h>

const int WIDTH = 1000;
const int HEIGHT = 1000;
const float FORCE = 01;
const float CENTRAL_MASS = 10;
const int LENGTH = 30000;
const float ZOOM = 100;

__global__ void update(float* d_positions, float* d_old_positions, Uint32* d_pixels){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= LENGTH){
        return;
    }
    
    float p1x = d_positions[i * 2];
    float p1y = d_positions[i * 2 + 1];

    float o1x = d_old_positions[i * 2];
    float o1y = d_old_positions[i * 2 + 1];
    
    float accel_x = 0;
    float accel_y = 0;
    
    for (int j = 0; j < LENGTH; j++){
        if (i == j){
            continue;
        }
        
        float p2x = d_positions[j * 2];
        float p2y = d_positions[j * 2 + 1];    
        
        float dx = p1x - p2x;
        float dy = p1y - p2y;
        
        float distance = dx * dx + dy * dy;
        
        float angle = atan2(dy, dx);
        
        float force = FORCE / distance;

        accel_x -= cos(angle) * force;
        accel_y -= sin(angle) * force;
    }

    float distance_ = p1x * p1x + p1y * p1y;
    float angle_ = atan2(p1y, p1x);

    //if (distance_ < 300) { return;}

    float force_ = FORCE * CENTRAL_MASS / distance_;

    accel_x -= cos(angle_) * force_;
    accel_y -= sin(angle_) * force_;

    p1x += accel_x;
    p1y += accel_y;
    
    float prev_x = p1x;
    float prev_y = p1y;

    d_positions[i * 2] = p1x * 2 - o1x;
    d_positions[i * 2 + 1] = p1y * 2 - o1y;

    d_old_positions[i * 2] = prev_x;
    d_old_positions[i * 2 + 1] = prev_y;

    int draw_x = (int)d_positions[i * 2] / ZOOM + WIDTH / 2;
    int draw_y = (int)d_positions[i * 2 + 1] / ZOOM + HEIGHT / 2;
    

    if (draw_x < 0 || draw_x >= WIDTH || draw_y < 0 || draw_y >= HEIGHT) {
        return;
    }
    
    Uint32 old_pixel = d_pixels[(draw_y) * WIDTH + (draw_x)];

    d_pixels[(draw_y) * WIDTH + (draw_x)] += 0x01010100 * 20;

    if (old_pixel > d_pixels[(draw_y) * WIDTH + (draw_x)]){
        d_pixels[(draw_y) * WIDTH + (draw_x)] = 0xFFFFFF00;
    }
    
    return;
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

float random(int Min, int Max)
{
    int diff = Max-Min;
    return (((float)(diff+1)/RAND_MAX) * rand() + Min);
}

int main(){
    dim3 threads(64);
    dim3 blocks((LENGTH + threads.x - 1) / threads.x);
    
    printf("%d blocks %d threads (%d threads; %d more than length (%d))\n", blocks.x, threads.x, blocks.x * threads.x, blocks.x * threads.x - LENGTH, LENGTH);

    float positions[LENGTH * 2];
    float old_positions[LENGTH * 2] = {0};
    Uint32 pixels[WIDTH * HEIGHT] = {0};

    for (int i = 0; i < LENGTH; i++){
        float value = random(100, 5000);
        positions[i * 2] = cos(i * (6.28319 / LENGTH)) * value;
        positions[i * 2 + 1] = sin(i * (6.28319 / LENGTH)) * value;
        old_positions[i * 2] = positions[i * 2] + sin(i * (6.28319 / LENGTH)) * value * 10 * (1 / value);
        old_positions[i * 2 + 1] = positions[i * 2 + 1] - cos(i * (6.28319 / LENGTH)) * value * 10 * (1 / value);
    }

    float* d_positions;
    float* d_old_positions;
    Uint32* d_pixels; 

    cudaMalloc(&d_positions, sizeof(float) * 2 * LENGTH);
    cudaMalloc(&d_old_positions, sizeof(float) * 2 * LENGTH);
    cudaMalloc(&d_pixels, sizeof(Uint32) * WIDTH * HEIGHT);

    cudaMemcpy(d_positions, positions, sizeof(float) * 2 * LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_positions, old_positions, sizeof(float) * 2 * LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixels, pixels, sizeof(Uint32) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);


    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("O(N^2) Gravity Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    SDL_Event event;

    double lastTime = getTime();

    int fps = 0;

    bool running = true;
    while (running){
        double currentTime = getTime();

        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
            }
        }

        memset(pixels, 0, WIDTH * HEIGHT * sizeof(Uint32));
        cudaMemcpy(d_pixels, pixels, sizeof(Uint32) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

        update<<<blocks, threads>>>(d_positions, d_old_positions, d_pixels);

        cudaMemcpy(pixels, d_pixels, sizeof(Uint32) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

        void* mPixels;
        int pitch;

        SDL_LockTexture(texture, NULL, &mPixels, &pitch);
        memcpy(mPixels, pixels, WIDTH * HEIGHT * sizeof(Uint32));
        SDL_UnlockTexture(texture);

        SDL_RenderCopy(renderer, texture, NULL, NULL);

        SDL_RenderPresent(renderer);

        double deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        fps = (deltaTime > 0) ? 1000.0f / deltaTime / 1000.0f : 0.0f;
        printf("%d fps\n", (int)fps);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    //free(positions);
    //free(old_positions);
    cudaFree(d_positions);
    cudaFree(d_old_positions);
    cudaFree(d_pixels);

    return 0;
}