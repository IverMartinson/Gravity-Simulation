#include <stdio.h>
#include <math.h>
#include <SDL2/SDL.h>

#define WIDTH 1000
#define HEIGHT 1000

const float GRAV_DIST = 500;
const float SCALE = 50;

float GRAV_SPEED = 0.01;

__global__ void distance_kernal(float* d_positions, float* d_velocities, Uint32* d_pixels, int length, float mouse_x, float mouse_y, bool using_mouse, float grav_angle){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length) return;

    float x = d_positions[i * 2];
    float y = d_positions[i * 2 + 1];
    float vx = d_velocities[i * 2];
    float vy = d_velocities[i * 2 + 1];

    float dx;
    float dy;
    
    if (using_mouse){
        dx = x - mouse_x;
        dy = y - mouse_y;
    }
    else {
        dx = x - cos(grav_angle) * GRAV_DIST;
        dy = y - sin(grav_angle) * GRAV_DIST;
    }

    float distance = sqrtf(dx * dx + dy * dy); 
    if (distance == 0) return;

    float angle = atan2f(dy, dx);
    float force = 10000.0f / distance;

    vx -= cosf(angle) * force;
    vy -= sinf(angle) * force;

    x += vx;
    y += vy;

    d_positions[i * 2] = x;
    d_positions[i * 2 + 1] = y;
    d_velocities[i * 2] = vx;
    d_velocities[i * 2 + 1] = vy;

    int x_ = (int)x / SCALE + 500;
    int y_ = (int)y / SCALE + 500;
    
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

bool using_mouse = true;

int main(){
    int length = 250'000'000;

    float* positions = (float*)malloc(length * 2 * sizeof(float));
    float* velocities = (float*)malloc(length * 2 * sizeof(float));

    for (int i = 0; i < length; i++) {
        positions[i * 2] = random(-50000, -1000);
        positions[i * 2 + 1] = random(-50000, -1000);
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

    int mouse_x, mouse_y;
    int w_o_x, w_o_y;

    float angle = 0;

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

        SDL_GetWindowPosition(window, &w_o_x, &w_o_y);

        SDL_GetGlobalMouseState(&mouse_x,&mouse_y);

        dim3 threads(64);
        dim3 blocks((length + threads.x - 1) / threads.x);

        printf("block: %dx, %dy  thread: %dx, %dy\n", blocks.x, blocks.y, threads.x, threads.y);
        
        memset(pixels, 0, WIDTH * HEIGHT * sizeof(Uint32));
        cudaMemcpy(d_pixels, pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyHostToDevice);

        mouse_x -= w_o_x;
        mouse_y -= w_o_y;

        angle += GRAV_SPEED;
        
        float d_mouse_x = (mouse_x - 500) * SCALE;
        float d_mouse_y = (mouse_y - 500) * SCALE;
        
        distance_kernal<<<blocks, threads>>>(d_positions, d_velocities, d_pixels, length, d_mouse_x, d_mouse_y, using_mouse, angle);
        
        cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);
        
        if (using_mouse && mouse_y * WIDTH + mouse_x < WIDTH * HEIGHT && mouse_y * WIDTH + mouse_x > 0){
            pixels[mouse_y * WIDTH + mouse_x] = 0xFFFFFFFF;
        }
        
        
                if (!using_mouse){for (int z = 0; z < 62; z += 1){
                    int x = (int)((cos(angle) * GRAV_DIST / SCALE + 500) + cos(z * 0.1) * 10);
                    int y = (int)((sin(angle) * GRAV_DIST / SCALE + 500) + sin(z * 0.1) * 10);
            
                    if (y * WIDTH + x < 0 || y * WIDTH + x >= WIDTH * HEIGHT) {
                        continue;
                    }
            
                    pixels[y * WIDTH + x] = 0xFFFFFFFF;
                }}
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
