/******************************************************************************
 * Progetto IN480 – Gaussian Blur (OpenMP) su immagine PPM con ROI
 *
 * Funzionalità:
 *   - Applica un Gaussian blur separabile (passata orizzontale + verticale).
 *   - La sfocatura viene applicata SOLO su una Regione di Interesse (ROI).
 *   - Supporta due modalità:
 *       1) --region  : ROI inserita da riga di comando (x y w h)
 *       2) --select  : selezione ROI interattiva con mouse (SDL2) su immagine PPM/PNG/JPG
 *
 * Input/Output:
 *   - Output sempre in formato PPM (P6).
 *   - Input consigliato PPM; la selezione interattiva supporta anche PNG/JPG.
 *
 * Requisiti:
 *   - GCC con supporto OpenMP
 *   - SDL2 (per --select)
 *
 * Installazione dipendenze (Ubuntu/Debian):
 *   sudo apt update
 *   sudo apt install build-essential libomp-dev libsdl2-dev
 *
 * Compilazione:
 *   gcc -std=c99 -Wall -Wextra -fopenmp -DHAVE_SDL2 gaussian_blur_omp.c roi_select.c stb_impl.c -o gaussian_blur_omp $(sdl2-config --cflags --libs) -lm
 *
 * Esecuzione:
 *   # ROI da riga di comando
 *   OMP_NUM_THREADS=8 ./gaussian_blur_omp --region input.ppm output.ppm x y w h
 *
 *   # ROI selezionata con il mouse (apre finestra)
 *   OMP_NUM_THREADS=8 ./gaussian_blur_omp --select input.ppm output.ppm
 *
 * Output tempi (stampa su stdout):
 *   Read:    tempo lettura immagine
 *   Compute: tempo calcolo blur sulla ROI
 *   Write:   tempo scrittura immagine
 *****************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <limits.h>

#ifdef HAVE_SDL2
  #include "roi_select.h"
#else
  /* fallback: compilazione senza SDL2 -> --select disabilitato */
  static int roi_select_interactive(const char* path, int* x, int* y, int* w, int* h) {
      (void)path; (void)x; (void)y; (void)w; (void)h;
      fprintf(stderr, "Errore: programma compilato senza SDL2, usa --region.\n");
      return 0;
  }
#endif

#define GAUSS_RADIUS 60

typedef struct { unsigned char r, g, b; } Pixel;

/* ---------------- prototipi funzioni ---------------- */
int clamp_int(int v, int lo, int hi);

void ppm_skip_comments(FILE *f);
int has_extension_ci(const char *path, const char *ext);

Pixel *ppm_read(const char *filepath, int *imgW, int *imgH);
void ppm_write(const char *filepath, const Pixel *buf, int imgW, int imgH);

Pixel *image_read_any(const char *inputPath, int *imgW, int *imgH);

float *gauss_kernel_1d_make(int R);

/* ---------------- utility ---------------- */

int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void ppm_skip_comments(FILE *f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') { while ((c = fgetc(f)) != '\n' && c != EOF) {} continue; }
        ungetc(c, f);
        break;
    }
}

int has_extension_ci(const char *path, const char *ext) {
    size_t lp = strlen(path), le = strlen(ext);
    if (lp < le) return 0;
    return strcasecmp(path + (lp - le), ext) == 0;
}

/* ---------------- PPM I/O ---------------- */

Pixel *ppm_read(const char *filepath, int *imgW, int *imgH) {
    FILE *f = fopen(filepath, "rb");
    if (!f) { perror("fopen"); return NULL; }

    char fmt[3] = {0};
    if (fscanf(f, "%2s", fmt) != 1) { fclose(f); return NULL; }
    ppm_skip_comments(f);

    if (fscanf(f, "%d %d", imgW, imgH) != 2) { fclose(f); return NULL; }
    ppm_skip_comments(f);

    int maxv = 0;
    if (fscanf(f, "%d", &maxv) != 1) { fclose(f); return NULL; }
    fgetc(f); /* whitespace */

    size_t n = (size_t)(*imgW) * (size_t)(*imgH);
    Pixel *buf = (Pixel*)malloc(n * sizeof(Pixel));
    if (!buf) { fclose(f); return NULL; }

    if (strcasecmp(fmt, "P6") == 0) {
        size_t got = fread(buf, sizeof(Pixel), n, f);
        if (got != n) { free(buf); fclose(f); return NULL; }

        if (maxv != 255 && maxv > 0) {
            for (size_t i = 0; i < n; i++) {
                buf[i].r = (unsigned char)((int)buf[i].r * 255 / maxv);
                buf[i].g = (unsigned char)((int)buf[i].g * 255 / maxv);
                buf[i].b = (unsigned char)((int)buf[i].b * 255 / maxv);
            }
        }
    } else {
        /* P3 ASCII */
        for (size_t i = 0; i < n; i++) {
            int r, g, b;
            if (fscanf(f, "%d %d %d", &r, &g, &b) != 3) {
                free(buf); fclose(f); return NULL;
            }
            buf[i].r = (unsigned char)clamp_int(r * 255 / maxv, 0, 255);
            buf[i].g = (unsigned char)clamp_int(g * 255 / maxv, 0, 255);
            buf[i].b = (unsigned char)clamp_int(b * 255 / maxv, 0, 255);
        }
    }

    fclose(f);
    return buf;
}

void ppm_write(const char *filepath, const Pixel *buf, int imgW, int imgH) {
    FILE *f = fopen(filepath, "wb");
    if (!f) { perror("fopen"); return; }
    fprintf(f, "P6\n%d %d\n255\n", imgW, imgH);
    fwrite(buf, sizeof(Pixel), (size_t)imgW * (size_t)imgH, f);
    fclose(f);
}

/* ---------------- input wrapper: PPM o convert ---------------- */

Pixel *image_read_any(const char *inputPath, int *imgW, int *imgH) {
    if (has_extension_ci(inputPath, ".ppm") || has_extension_ci(inputPath, ".pnm")) {
        return ppm_read(inputPath, imgW, imgH);
    }

    /* convert -> PPM temporaneo */
    char tmpPath[PATH_MAX];
    char cmd[PATH_MAX * 2];

    snprintf(tmpPath, sizeof(tmpPath), "/tmp/gaussblur_%d.ppm", (int)getpid());
    snprintf(cmd, sizeof(cmd), "convert \"%s\" -compress none \"%s\"", inputPath, tmpPath);

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Errore: convert fallita (serve ImageMagick)\n");
        return NULL;
    }

    Pixel *img = ppm_read(tmpPath, imgW, imgH);
    remove(tmpPath);

    if (!img) fprintf(stderr, "Errore lettura PPM temporaneo\n");
    return img;
}

/* ---------------- kernel gaussiano 1D ---------------- */

float *gauss_kernel_1d_make(int R) {
    const int K = 2 * R + 1;
    const float sigma = (R > 0) ? (0.5f * (float)R) : 1.0f;
    const float twoSigma2 = 2.0f * sigma * sigma;

    float *k = (float*)malloc((size_t)K * sizeof(float));
    if (!k) return NULL;

    float sum = 0.0f;
    for (int i = -R; i <= R; i++) {
        float v = expf(-(i * (float)i) / twoSigma2);
        k[i + R] = v;
        sum += v;
    }
    for (int i = 0; i < K; i++) k[i] /= sum;
    return k;
}

/* ---------------- MAIN ---------------- */

int main(int argc, char **argv) {
    
    
    const char *inputPath  = NULL;
    const char *outputPath = NULL;
    int roiX = 0, roiY = 0, roiW = 0, roiH = 0;

    // 1) Modalità: --select input output
    if (argc == 4 && strcmp(argv[1], "--select") == 0) {

        inputPath  = argv[2];
        outputPath = argv[3];

        if (!roi_select_interactive(inputPath, &roiX, &roiY, &roiW, &roiH)) {
            fprintf(stderr, "Selezione annullata.\n");
            return 1;
        }

    // 2) Modalità: --region input output x y w h
    } else if (argc == 8 && strcmp(argv[1], "--region") == 0) {

        inputPath  = argv[2];
        outputPath = argv[3];

        roiX = atoi(argv[4]);
        roiY = atoi(argv[5]);
        roiW = atoi(argv[6]);
        roiH = atoi(argv[7]);

    // 3) Modalità “runtime” (opzionale): nessun argomento → chiedi input/output e poi selezioni
    } else if (argc == 1) {

        static char inBuf[512];
        static char outBuf[512];

        printf("Inserisci path immagine: ");
        fflush(stdout);
        if (!fgets(inBuf, sizeof(inBuf), stdin)) return 1;
        inBuf[strcspn(inBuf, "\n")] = 0;

        printf("Inserisci output (es. out.ppm): ");
        fflush(stdout);
        if (!fgets(outBuf, sizeof(outBuf), stdin)) return 1;
        outBuf[strcspn(outBuf, "\n")] = 0;

        inputPath  = inBuf;
        outputPath = outBuf;

        if (!roi_select_interactive(inputPath, &roiX, &roiY, &roiW, &roiH)) {
            fprintf(stderr, "Selezione annullata.\n");
            return 1;
        }

    } else {
        fprintf(stderr,
            "Uso:\n"
            "  %s --region input.(ppm|png|jpg|...) output.ppm x y w h\n"
            "  %s --select input.(ppm|png|jpg|...) output.ppm\n"
            "  %s            (senza argomenti: ti chiede input/output e poi selezioni)\n",
            argv[0], argv[0], argv[0]
        );
        return 1;
    }

    int imgW = 0, imgH = 0;

    double tRead0 = omp_get_wtime();
    Pixel *img = image_read_any(inputPath, &imgW, &imgH);
    double tRead1 = omp_get_wtime();

    if (!img) { fprintf(stderr, "Errore lettura immagine\n"); return 1; }

    if (roiX < 0) roiX = 0;
    if (roiY < 0) roiY = 0;
    if (roiX >= imgW || roiY >= imgH) {
        fprintf(stderr, "ROI fuori immagine\n");
        free(img);
        return 1;
    }
    if (roiX + roiW > imgW) roiW = imgW - roiX;
    if (roiY + roiH > imgH) roiH = imgH - roiY;
    if (roiW <= 0 || roiH <= 0) { fprintf(stderr, "ROI vuota\n"); free(img); return 1; }

    const int R = GAUSS_RADIUS;
    const int K = 2 * R + 1;
    const int paddedW = roiW + 2 * R;
    const int paddedH = roiH + 2 * R;

    Pixel *out = (Pixel*)malloc((size_t)imgW * (size_t)imgH * sizeof(Pixel));
    float *kernel = gauss_kernel_1d_make(R);
    Pixel *paddedROI = (Pixel*)malloc((size_t)paddedW * (size_t)paddedH * sizeof(Pixel));
    Pixel *tmpT      = (Pixel*)malloc((size_t)roiW * (size_t)roiH * sizeof(Pixel)); /* trasposto */

    if (!out || !kernel || !paddedROI || !tmpT) {
        fprintf(stderr, "Alloc fallita\n");
        free(img); free(out); free(kernel); free(paddedROI); free(tmpT);
        return 1;
    }

    memcpy(out, img, (size_t)imgW * (size_t)imgH * sizeof(Pixel));

    double tComp0 = omp_get_wtime();

    #pragma omp parallel default(none) \
        shared(img, out, kernel, paddedROI, tmpT, imgW, imgH, roiX, roiY, roiW, roiH) \
        firstprivate(R, K, paddedW, paddedH)
    {
        #pragma omp for schedule(static)
        for (int py = 0; py < paddedH; py++) {
            int srcY = clamp_int(roiY + py - R, 0, imgH - 1);
            for (int px = 0; px < paddedW; px++) {
                int srcX = clamp_int(roiX + px - R, 0, imgW - 1);
                paddedROI[(size_t)py * (size_t)paddedW + (size_t)px] =
                    img[(size_t)srcY * (size_t)imgW + (size_t)srcX];
            }
        }

        #pragma omp for schedule(static)
        for (int y = 0; y < roiH; y++) {
            const Pixel *padRow = paddedROI + (size_t)(y + R) * (size_t)paddedW;
            for (int x = 0; x < roiW; x++) {
                float accR = 0.f, accG = 0.f, accB = 0.f;
                const Pixel *win = padRow + x;

                for (int t = 0; t < K; t++) {
                    float w = kernel[t];
                    accR += w * win[t].r;
                    accG += w * win[t].g;
                    accB += w * win[t].b;
                }

                tmpT[(size_t)x * (size_t)roiH + (size_t)y] = (Pixel){
                    (unsigned char)(accR + 0.5f),
                    (unsigned char)(accG + 0.5f),
                    (unsigned char)(accB + 0.5f)
                };
            }
        }

        #pragma omp for schedule(static)
        for (int x = 0; x < roiW; x++) {
            const Pixel *col = tmpT + (size_t)x * (size_t)roiH;
            for (int y = 0; y < roiH; y++) {
                float accR = 0.f, accG = 0.f, accB = 0.f;
                for (int dt = -R; dt <= R; dt++) {
                    int yy = clamp_int(y + dt, 0, roiH - 1);
                    Pixel p = col[yy];
                    float w = kernel[dt + R];
                    accR += w * p.r;
                    accG += w * p.g;
                    accB += w * p.b;
                }
                out[(size_t)(roiY + y) * (size_t)imgW + (size_t)(roiX + x)] = (Pixel){
                    (unsigned char)(accR + 0.5f),
                    (unsigned char)(accG + 0.5f),
                    (unsigned char)(accB + 0.5f)
                };
            }
        }
    }

    double tComp1 = omp_get_wtime();

    double tWrite0 = omp_get_wtime();
    ppm_write(outputPath, out, imgW, imgH);
    double tWrite1 = omp_get_wtime();

    printf("Read:    %.6f s\n", tRead1  - tRead0);
    printf("Compute: %.6f s\n", tComp1  - tComp0);
    printf("Write:   %.6f s\n", tWrite1 - tWrite0);

    free(img); free(out); free(kernel); free(paddedROI); free(tmpT);
    return 0;
}

