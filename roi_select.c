#include "roi_select.h"

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

#include "stb_image.h"

/* -------------------- helpers -------------------- */

static int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int has_ext_ci(const char *path, const char *ext) {
    size_t lp = strlen(path), le = strlen(ext);
    if (lp < le) return 0;
    return strcasecmp(path + (lp - le), ext) == 0;
}

static void ppm_skip_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(f)) != '\n' && c != EOF) {}
            continue;
        }
        ungetc(c, f);
        break;
    }
}

/* Legge PPM P6 o P3 e restituisce buffer RGB 8-bit (3*W*H). */
static unsigned char* ppm_read_rgb8(const char *path, int *W, int *H) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    char fmt[3] = {0,0,0};
    if (fscanf(f, "%2s", fmt) != 1) { fclose(f); return 0; }

    ppm_skip_comments(f);
    if (fscanf(f, "%d %d", W, H) != 2) { fclose(f); return 0; }

    ppm_skip_comments(f);
    int maxv = 0;
    if (fscanf(f, "%d", &maxv) != 1) { fclose(f); return 0; }

    /* consuma 1 whitespace dopo maxv */
    (void)fgetc(f);

    if (*W <= 0 || *H <= 0 || maxv <= 0) { fclose(f); return 0; }

    size_t n = (size_t)(*W) * (size_t)(*H);
    unsigned char *rgb = (unsigned char*)malloc(3 * n);
    if (!rgb) { fclose(f); return 0; }

    if (strcasecmp(fmt, "P6") == 0) {
        /* P6 binario */
        if (fread(rgb, 3, n, f) != n) 
        { 
        free(rgb); 
        fclose(f); return 0; }

        if (maxv != 255) {
            for (size_t i = 0; i < 3*n; i++) {
                rgb[i] = (unsigned char)((int)rgb[i] * 255 / maxv);
            }
        }

    } else if (strcasecmp(fmt, "P3") == 0) {
        /* P3 ASCII */
        for (size_t i = 0; i < n; i++) {
            int r,g,b;
            if (fscanf(f, "%d %d %d", &r, &g, &b) != 3) {
                free(rgb); fclose(f); return 0;
            }
            if (maxv != 255) {
                r = r * 255 / maxv;
                g = g * 255 / maxv;
                b = b * 255 / maxv;
            }
            if (r < 0) r = 0; 
            if (r > 255) r = 255;
            if (g < 0) g = 0; 
            if (g > 255) g = 255;
            if (b < 0) b = 0; 
            if (b > 255) b = 255;

            rgb[3*i + 0] = (unsigned char)r;
            rgb[3*i + 1] = (unsigned char)g;
            rgb[3*i + 2] = (unsigned char)b;
        }
    } else {
        free(rgb);
        fclose(f);
        return 0;
    }

    fclose(f);
    return rgb;
}

/* RGB -> RGBA (alpha=255) */
static unsigned char* rgb_to_rgba(const unsigned char *rgb, int W, int H) {
    size_t n = (size_t)W * (size_t)H;
    unsigned char *rgba = (unsigned char*)malloc(4 * n);
    if (!rgba) return 0;
    for (size_t i = 0; i < n; i++) {
        rgba[4*i + 0] = rgb[3*i + 0];
        rgba[4*i + 1] = rgb[3*i + 1];
        rgba[4*i + 2] = rgb[3*i + 2];
        rgba[4*i + 3] = 255;
    }
    return rgba;
}

/* -------------------- public API -------------------- */

int roi_select_interactive(const char *image_path,
                           int *out_x, int *out_y, int *out_w, int *out_h)
{
    if (!image_path || !out_x || !out_y || !out_w || !out_h) return 0;

    int img_w = 0, img_h = 0;
    unsigned char *pixels_rgba = 0;
    int from_stb = 0; /* 1 => libera con stbi_image_free, 0 => free */

    /* --- LOAD --- */
    if (has_ext_ci(image_path, ".ppm") || has_ext_ci(image_path, ".pnm")) {
        unsigned char *rgb = ppm_read_rgb8(image_path, &img_w, &img_h);
        if (!rgb) {
            fprintf(stderr, "ppm_read_rgb8 failed: can't read '%s'\n", image_path);
            return 0;
        }
        pixels_rgba = rgb_to_rgba(rgb, img_w, img_h);
        free(rgb);

        if (!pixels_rgba) {
            fprintf(stderr, "rgb_to_rgba failed\n");
            return 0;
        }
        from_stb = 0;
    } else {
        int img_c = 0;
        pixels_rgba = stbi_load(image_path, &img_w, &img_h, &img_c, 4);
        if (!pixels_rgba) {
            fprintf(stderr, "stbi_load failed: %s\n", stbi_failure_reason());
            return 0;
        }
        from_stb = 1;
    }

    /* --- SDL INIT --- */
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        if (from_stb) stbi_image_free(pixels_rgba); else free(pixels_rgba);
        return 0;
    }

    SDL_Window *win = SDL_CreateWindow(
        "ROI Select - drag mouse | ENTER=OK | ESC=Cancel",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        img_w, img_h,
        SDL_WINDOW_SHOWN
    );
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        if (from_stb) stbi_image_free(pixels_rgba); else free(pixels_rgba);
        return 0;
    }

    SDL_Renderer *ren = SDL_CreateRenderer(
        win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!ren) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        if (from_stb) stbi_image_free(pixels_rgba); else free(pixels_rgba);
        return 0;
    }

    SDL_Texture *tex = SDL_CreateTexture(
        ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, img_w, img_h
    );
    if (!tex) {
        fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        SDL_Quit();
        if (from_stb) stbi_image_free(pixels_rgba); else free(pixels_rgba);
        return 0;
    }

    SDL_UpdateTexture(tex, 0, pixels_rgba, img_w * 4);

    /* --- ROI selection loop --- */
    int selecting = 0;
    int have_roi  = 0;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;

    int running = 1;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                have_roi = 0;
                running = 0;
            } else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    have_roi = 0;
                    running = 0;
                } else if (e.key.keysym.sym == SDLK_RETURN || e.key.keysym.sym == SDLK_KP_ENTER) {
                    if (have_roi) running = 0;
                }
            } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                selecting = 1;
                x0 = x1 = clampi(e.button.x, 0, img_w - 1);
                y0 = y1 = clampi(e.button.y, 0, img_h - 1);
                have_roi = 0;
            } else if (e.type == SDL_MOUSEMOTION && selecting) {
                x1 = clampi(e.motion.x, 0, img_w - 1);
                y1 = clampi(e.motion.y, 0, img_h - 1);
            } else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                selecting = 0;

                int left   = (x0 < x1) ? x0 : x1;
                int right  = (x0 > x1) ? x0 : x1;
                int top    = (y0 < y1) ? y0 : y1;
                int bottom = (y0 > y1) ? y0 : y1;

                int w = right - left + 1;
                int h = bottom - top + 1;

                if (w >= 2 && h >= 2) {
                    x0 = left;  y0 = top;
                    x1 = right; y1 = bottom;
                    have_roi = 1;

                    fprintf(stdout, "ROI: x=%d y=%d w=%d h=%d\n", x0, y0, w, h);
                    fflush(stdout);
                } else {
                    have_roi = 0;
                }
            }
        }

        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, 0, 0);

        if (selecting || have_roi) {
            int left   = (x0 < x1) ? x0 : x1;
            int right  = (x0 > x1) ? x0 : x1;
            int top    = (y0 < y1) ? y0 : y1;
            int bottom = (y0 > y1) ? y0 : y1;

            SDL_Rect r;
            r.x = left;
            r.y = top;
            r.w = right - left + 1;
            r.h = bottom - top + 1;

            SDL_SetRenderDrawColor(ren, 255, 0, 0, 255);
            SDL_RenderDrawRect(ren, &r);
            SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        }

        SDL_RenderPresent(ren);
    }

    int ok = 0;
    if (have_roi) {
        *out_x = x0;
        *out_y = y0;
        *out_w = x1 - x0 + 1;
        *out_h = y1 - y0 + 1;
        ok = 1;
    }

    /* --- cleanup --- */
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();

    if (from_stb) stbi_image_free(pixels_rgba);
    else free(pixels_rgba);

    return ok;
}
