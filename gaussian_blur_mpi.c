/******************************************************************************
 * Progetto IN480 – Gaussian Blur (MPI) su immagine PPM con ROI
 *
 * Funzionalità:
 *   - Applica un Gaussian blur separabile (passata orizzontale + verticale).
 *   - La sfocatura viene applicata SOLO su una Regione di Interesse (ROI).
 *   - Supporta due modalità:
 *       1) --region  : ROI inserita da riga di comando (x y w h)
 *       2) --select  : selezione ROI interattiva con mouse (SDL2) su immagine PPM/PNG/JPG
 *
 * Strategia MPI (riassunto):
 *   - rank 0 gestisce parsing argomenti e (se richiesto) selezione ROI con finestra.
 *   - rank 0 fa broadcast a tutti di inputPath/outputPath e ROI (x0,y0,w,h).
 *   - La ROI viene divisa tra i processi (es. per blocchi di colonne o righe).
 *   - Ogni processo calcola la sua porzione e poi invia il risultato a rank 0 (gather).
 *   - rank 0 ricompone l’immagine finale e salva output PPM.
 *
 * Requisiti:
 *   - Implementazione MPI
 *   - SDL2 (per --select)
 *
 * Installazione dipendenze (Ubuntu/Debian):
 *   sudo apt install build-essential mpi-default-bin mpi-default-dev libsdl2-dev
 *
 * Compilazione:
 *   mpicc -std=c99 -Wall -Wextra -DHAVE_SDL2 gaussian_blur_mpi.c roi_select.c stb_impl.c -o gaussian_blur_mpi $(sdl2-config --cflags --libs) -lm
 *
 * Esecuzione:
 *   # ROI da riga di comando
 *   mpirun -np 4 ./gaussian_blur_mpi --region input.ppm output.ppm x y w h
 *
 *   # ROI selezionata con il mouse (solo rank 0 apre la finestra)
 *   mpirun -np 4 ./gaussian_blur_mpi --select input.ppm output.ppm
 *
 * Output tempi (stampa su stdout, SOLO rank 0):
 *   Read:    tempo lettura immagine (rank 0)
 *   Compute: tempo calcolo massimo tra i processi (MPI_Reduce max)
 *   Write:   tempo scrittura immagine (rank 0)
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <mpi.h>
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

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define R 60   /* raggio kernel gaussiano (2*R+1) */

typedef struct { unsigned char r, g, b; } Pixel;

/* ---------------- prototipi funzioni ---------------- */

int clamp_int(int v, int lo, int hi);

void ppm_skip_comments(FILE *f);
int has_extension_ci(const char *path, const char *ext);

Pixel *ppm_read(const char *filepath, int *imgW, int *imgH);
void ppm_write(const char *filepath, const Pixel *buf, int imgW, int imgH);

Pixel *image_read_any(const char *inputPath, int *imgW, int *imgH);

float *make_gauss1d(void);

/* ---------------- utility (uguale a OpenMP) ---------------- */

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

/* ---------------- PPM I/O (uguale a OpenMP) ---------------- */

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
            if (maxv > 0) {
                r = r * 255 / maxv;
                g = g * 255 / maxv;
                b = b * 255 / maxv;
            }
            buf[i].r = (unsigned char)clamp_int(r, 0, 255);
            buf[i].g = (unsigned char)clamp_int(g, 0, 255);
            buf[i].b = (unsigned char)clamp_int(b, 0, 255);
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

/* ---------------- input wrapper: PPM o convert (uguale a OpenMP) ---------------- */

Pixel *image_read_any(const char *inputPath, int *imgW, int *imgH) {
    if (has_extension_ci(inputPath, ".ppm") || has_extension_ci(inputPath, ".pnm")) {
        return ppm_read(inputPath, imgW, imgH);
    }

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

float* make_gauss1d(void) {
    const int K = 2 * R + 1;
    const float sigma = (R > 0) ? (0.5f * (float)R) : 1.0f;
    const float two = 2.0f * sigma * sigma;

    float* k = (float*)malloc((size_t)K * sizeof(float));
    if (!k) return NULL;

    float sum = 0.f;
    for (int i = -R; i <= R; i++) {
        float v = expf(-(i * (float)i) / two);
        k[i + R] = v;
        sum += v;
    }
    for (int i = 0; i < K; i++) k[i] /= sum;
    return k;
}

/* ========================== MAIN =================================== */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  /* avvia ambiente MPI */

    /* Identità del processo e numero totale di processi */
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    /* rank = id del processo */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);  /* nprocs = numero processi totali */

    //fprintf(stderr, "DEBUG: pid=%d rank=%d nprocs=%d\n", (int)getpid(), rank, nprocs);
    //fflush(stderr);

    /* Buffer path: servono perché i processi MPI non condividono argv/memoria */
    char inBuf[PATH_MAX]  = {0};
    char outBuf[PATH_MAX] = {0};

    const char* inpath  = inBuf;
    const char* outpath = outBuf;

    /* ROI globale (verrà condivisa a tutti) */
    int x0 = 0, y0 = 0, rw = 0, rh = 0;

    if (rank == 0) {
        /* Solo rank 0 fa parsing argomenti e seleziona ROI (eventuale SDL) */
        if (argc == 4 && strcmp(argv[1], "--select") == 0) {
            // --select input output
            snprintf(inBuf,  sizeof(inBuf),  "%s", argv[2]);
            snprintf(outBuf, sizeof(outBuf), "%s", argv[3]);

            if (!roi_select_interactive(inBuf, &x0, &y0, &rw, &rh)) {
                fprintf(stderr, "Selezione annullata.\n");
                rw = rh = 0; // segnala errore
            }

        } else if (argc == 8 && strcmp(argv[1], "--region") == 0) {
            // --region input output x y w h
            snprintf(inBuf,  sizeof(inBuf),  "%s", argv[2]);
            snprintf(outBuf, sizeof(outBuf), "%s", argv[3]);

            x0 = atoi(argv[4]);
            y0 = atoi(argv[5]);
            rw = atoi(argv[6]);
            rh = atoi(argv[7]);

        } else {
            fprintf(stderr,
                "Uso:\n"
                "  %s --region input.(ppm|png|jpg|jpeg) output.ppm x y w h\n"
                "  %s --select input.(ppm|png|jpg|jpeg) output.ppm\n",
                argv[0], argv[0]
            );
            rw = rh = 0;
        }
    }

    /* MPI_Bcast: il rank 0 distribuisce a tutti path e ROI (stesse info per tutti) */
    MPI_Bcast(inBuf,  (int)sizeof(inBuf),  MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(outBuf, (int)sizeof(outBuf), MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Bcast(&x0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rw, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rh, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* se errore/annullato, escono tutti (stessa condizione su ogni rank) */
    if (rw <= 0 || rh <= 0) {
        MPI_Finalize();  /* chiusura MPI pulita */
        return 1;
    }

    int W = 0, H = 0;
    Pixel* img = NULL;

    double tR0 = 0.0, tR1 = 0.0;

    if (rank == 0) {
        /* Solo rank 0 legge l’immagine dal disco */
        tR0 = MPI_Wtime();
        img = image_read_any(inpath, &W, &H);
        tR1 = MPI_Wtime();

        if (!img) {
            fprintf(stderr, "Errore lettura immagine\n");
            MPI_Abort(MPI_COMM_WORLD, 2); /* termina TUTTI i processi in errore */
        }

        /* clamp ROI dentro l'immagine */
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x0 >= W || y0 >= H) {
            fprintf(stderr, "ROI fuori immagine\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        if (x0 + rw > W) rw = W - x0;
        if (y0 + rh > H) rh = H - y0;
        if (rw <= 0 || rh <= 0) {
            fprintf(stderr, "ROI vuota\n");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    /* MPI_Bcast: condivido a tutti i metadati (W,H) e la ROI “corretta” clamped */
    MPI_Bcast(&W,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rw, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rh, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* MPI Datatype: definisco un tipo che rappresenta un Pixel (3 byte) */
    MPI_Datatype MPI_PIXEL;
    MPI_Type_contiguous((int)sizeof(Pixel), MPI_BYTE, &MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);

    /* kernel gaussiano (uguale su tutti) */
    float* k = make_gauss1d();
    if (!k) {
        if (rank == 0) free(img);
        MPI_Type_free(&MPI_PIXEL);
        MPI_Finalize();
        return 1;
    }

    /* decomposizione 1D in X della ROI: ogni processo riceve un blocco di colonne */
    int base_w = rw / nprocs;
    int rem_w  = rw % nprocs;

    int my_w = base_w + (rank < rem_w ? 1 : 0);

    int my_offx;
    if (rank < rem_w) {
        my_offx = rank * (base_w + 1);
    } else {
        my_offx = rem_w * (base_w + 1) + (rank - rem_w) * base_w;
    }

    int my_h = rh;

    Pixel* local_in  = 0;  /* sotto-ROI locale in input */
    Pixel* local_out = 0;  /* sotto-ROI locale in output */

    /* Scatter manuale: rank 0 invia ad ogni rank il suo blocco di ROI */
    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            int lw = base_w + (r < rem_w ? 1 : 0);
            if (lw <= 0) continue;

            int offx;
            if (r < rem_w) {
                offx = r * (base_w + 1);
            } else {
                offx = rem_w * (base_w + 1) + (r - rem_w) * base_w;
            }

            if (r == 0) {
                /* rank 0: copia direttamente la sua parte in local_in */
                local_in = (Pixel*)malloc((size_t)my_h * (size_t)my_w * sizeof(Pixel));
                if (!local_in) MPI_Abort(MPI_COMM_WORLD, 5);

                for (int yy = 0; yy < my_h; ++yy) {
                    memcpy(local_in + (size_t)yy * (size_t)my_w,
                           img + (size_t)(y0 + yy) * (size_t)W + (size_t)(x0 + offx),
                           (size_t)my_w * sizeof(Pixel));
                }
            } else {
                /* altri rank: preparo un buffer tmp e lo invio con MPI_Send */
                Pixel* tmp = (Pixel*)malloc((size_t)my_h * (size_t)lw * sizeof(Pixel));
                if (!tmp) MPI_Abort(MPI_COMM_WORLD, 5);

                for (int yy = 0; yy < my_h; ++yy) {
                    memcpy(tmp + (size_t)yy * (size_t)lw,
                           img + (size_t)(y0 + yy) * (size_t)W + (size_t)(x0 + offx),
                           (size_t)lw * sizeof(Pixel));
                }

                MPI_Send(tmp, my_h * lw, MPI_PIXEL, r, 10, MPI_COMM_WORLD); /* invio blocco ROI */
                free(tmp);
            }
        }
    } else {
        if (my_w > 0) {
            /* ogni rank != 0 riceve il proprio blocco ROI dal rank 0 */
            local_in = (Pixel*)malloc((size_t)my_h * (size_t)my_w * sizeof(Pixel));
            if (!local_in) MPI_Abort(MPI_COMM_WORLD, 5);

            MPI_Recv(local_in, my_h * my_w, MPI_PIXEL, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (my_w > 0) {
        local_out = (Pixel*)malloc((size_t)my_h * (size_t)my_w * sizeof(Pixel));
        if (!local_out) MPI_Abort(MPI_COMM_WORLD, 6);
    }

    MPI_Barrier(MPI_COMM_WORLD);         /* sincronizzazione: inizio compute */
    double tC0 = MPI_Wtime();

    if (my_w > 0) {

        /* Halo exchange: scambio colonne di bordo con i vicini per la passata orizzontale */
        const int halo = (R < my_w ? R : my_w);   /* halo effettivo = min(R, my_w) */
        const int has_left  = (rank > 0);
        const int has_right = (rank < nprocs - 1);
        const int use_halo  = (halo > 0) && (has_left || has_right);

        Pixel* left_buf  = NULL;   /* colonne ricevute dal vicino sinistro */
        Pixel* right_buf = NULL;   /* colonne ricevute dal vicino destro */

        if (use_halo) {
            /* Datatype che descrive una striscia di 'halo' colonne in una matrice row-major */
            MPI_Datatype strip_type;
            MPI_Type_vector(my_h, halo, my_w, MPI_PIXEL, &strip_type);
            MPI_Type_commit(&strip_type);

            if (has_left)  left_buf  = (Pixel*)malloc((size_t)my_h * (size_t)halo * sizeof(Pixel));
            if (has_right) right_buf = (Pixel*)malloc((size_t)my_h * (size_t)halo * sizeof(Pixel));
            if ((has_left && !left_buf) || (has_right && !right_buf)) MPI_Abort(MPI_COMM_WORLD, 7);

            /* Request: handle per le operazioni non bloccanti (fino a 4: 2 recv + 2 send) */
            MPI_Request reqs[4];
            int rq = 0;

            /* Avvio ricezioni non bloccanti (Irecv) */
            if (has_left)
                MPI_Irecv(left_buf,  my_h * halo, MPI_PIXEL, rank - 1, 20, MPI_COMM_WORLD, &reqs[rq++]);
            if (has_right)
                MPI_Irecv(right_buf, my_h * halo, MPI_PIXEL, rank + 1, 21, MPI_COMM_WORLD, &reqs[rq++]);

            /* Avvio invii non bloccanti (Isend) delle prime/ultime 'halo' colonne */
            if (has_left) {
                Pixel* send_left = local_in; /* prime halo colonne */
                MPI_Isend(send_left, 1, strip_type, rank - 1, 21, MPI_COMM_WORLD, &reqs[rq++]);
            }
            if (has_right) {
                Pixel* send_right = local_in + (my_w - halo); /* ultime halo colonne */
                MPI_Isend(send_right, 1, strip_type, rank + 1, 20, MPI_COMM_WORLD, &reqs[rq++]);
            }

            /* Aspetto che tutte le comunicazioni halo siano complete */
            MPI_Waitall(rq, reqs, MPI_STATUSES_IGNORE);

            MPI_Type_free(&strip_type);
        }

        /* Calcolo locale: padding + passata orizzontale + passata verticale */
        const int pad_w = my_w + 2 * R;
        Pixel* pad = (Pixel*)malloc((size_t)my_h * (size_t)pad_w * sizeof(Pixel));
        Pixel* tmp = (Pixel*)malloc((size_t)my_h * (size_t)my_w * sizeof(Pixel));
        if (!pad || !tmp) MPI_Abort(MPI_COMM_WORLD, 8);

        for (int yy = 0; yy < my_h; ++yy) {
            Pixel* prow = pad + (size_t)yy * (size_t)pad_w;
            Pixel* src  = local_in + (size_t)yy * (size_t)my_w;

            /* centro: copio i pixel locali */
            memcpy(prow + R, src, (size_t)my_w * sizeof(Pixel));

            /* bordo sinistro: replica o usa halo ricevuto */
            Pixel edgeL = src[0];
            for (int i = 0; i < R; ++i) prow[i] = edgeL;
            if (has_left && halo > 0 && left_buf) {
                Pixel* lb = left_buf + (size_t)yy * (size_t)halo;
                memcpy(prow + (R - halo), lb, (size_t)halo * sizeof(Pixel));
            }

            /* bordo destro: replica o usa halo ricevuto */
            Pixel edgeR = src[my_w - 1];
            for (int i = 0; i < R; ++i) prow[R + my_w + i] = edgeR;
            if (has_right && halo > 0 && right_buf) {
                Pixel* rb = right_buf + (size_t)yy * (size_t)halo;
                memcpy(prow + (R + my_w), rb, (size_t)halo * sizeof(Pixel));
            }
        }

        if (left_buf)  free(left_buf);
        if (right_buf) free(right_buf);

        /* orizzontale: pad -> tmp */
        for (int yy = 0; yy < my_h; ++yy) {
            const Pixel* row = pad + (size_t)yy * (size_t)pad_w;
            for (int xx = 0; xx < my_w; ++xx) {
                float ar = 0.f, ag = 0.f, ab = 0.f;
                const Pixel* win = row + xx;
                for (int t = 0; t < 2 * R + 1; ++t) {
                    float wt = k[t];
                    ar += wt * win[t].r;
                    ag += wt * win[t].g;
                    ab += wt * win[t].b;
                }
                tmp[(size_t)yy * (size_t)my_w + (size_t)xx] = (Pixel){
                    (unsigned char)(ar + 0.5f),
                    (unsigned char)(ag + 0.5f),
                    (unsigned char)(ab + 0.5f)
                };
            }
        }

        /* verticale: tmp -> local_out (solo clamp in Y perché la decomposizione è in X) */
        for (int xx = 0; xx < my_w; ++xx) {
            for (int yy = 0; yy < my_h; ++yy) {
                float ar = 0.f, ag = 0.f, ab = 0.f;
                for (int t = -R; t <= R; ++t) {
                    int yi = yy + t;
                    if (yi < 0) yi = 0;
                    if (yi >= my_h) yi = my_h - 1;
                    Pixel p = tmp[(size_t)yi * (size_t)my_w + (size_t)xx];
                    float wt = k[t + R];
                    ar += wt * p.r;
                    ag += wt * p.g;
                    ab += wt * p.b;
                }
                local_out[(size_t)yy * (size_t)my_w + (size_t)xx] = (Pixel){
                    (unsigned char)(ar + 0.5f),
                    (unsigned char)(ag + 0.5f),
                    (unsigned char)(ab + 0.5f)
                };
            }
        }

        free(pad);
        free(tmp);
    }

    MPI_Barrier(MPI_COMM_WORLD);         /* sincronizzazione: fine compute */
    double tC1 = MPI_Wtime();

    double my_comp = tC1 - tC0;

    /* MPI_Reduce: rank 0 prende il massimo dei tempi (tempo reale della fase parallela) */
    double comp_max = 0.0;
    MPI_Reduce(&my_comp, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Gather manuale: ogni rank invia al rank 0 il blocco sfocato */
    Pixel* out_full = NULL;

    if (rank == 0) {
        out_full = (Pixel*)malloc((size_t)W * (size_t)H * sizeof(Pixel));
        if (!out_full) MPI_Abort(MPI_COMM_WORLD, 9);

        /* inizializzo l’output copiando l’immagine originale (fuori ROI resta uguale) */
        memcpy(out_full, img, (size_t)W * (size_t)H * sizeof(Pixel));

        /* rank 0 copia il proprio blocco già calcolato */
        if (my_w > 0) {
            for (int yy = 0; yy < my_h; ++yy) {
                memcpy(out_full + (size_t)(y0 + yy) * (size_t)W + (size_t)(x0 + my_offx),
                       local_out + (size_t)yy * (size_t)my_w,
                       (size_t)my_w * sizeof(Pixel));
            }
        }

        /* ricevo i blocchi dagli altri rank e li inserisco nella posizione corretta */
        for (int r = 1; r < nprocs; ++r) {
            int lw = base_w + (r < rem_w ? 1 : 0);
            if (lw <= 0) continue;

            int offx;
            if (r < rem_w) {
                offx = r * (base_w + 1);
            } else {
                offx = rem_w * (base_w + 1) + (r - rem_w) * base_w;
            }

            Pixel* tmpblk = (Pixel*)malloc((size_t)my_h * (size_t)lw * sizeof(Pixel));
            if (!tmpblk) MPI_Abort(MPI_COMM_WORLD, 10);

            MPI_Recv(tmpblk, my_h * lw, MPI_PIXEL, r, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int yy = 0; yy < my_h; ++yy) {
                memcpy(out_full + (size_t)(y0 + yy) * (size_t)W + (size_t)(x0 + offx),
                       tmpblk + (size_t)yy * (size_t)lw,
                       (size_t)lw * sizeof(Pixel));
            }
            free(tmpblk);
        }
    } else {
        /* rank != 0 invia il proprio blocco al master */
        if (my_w > 0) {
            MPI_Send(local_out, my_h * my_w, MPI_PIXEL, 0, 30, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); /* tutti hanno finito gather */

    if (rank == 0) {
        /* Solo rank 0 salva e stampa i tempi */
        double tW0 = MPI_Wtime();
        ppm_write(outpath, out_full, W, H);
        double tW1 = MPI_Wtime();

        printf("Read:    %.6f s\n", tR1 - tR0);
        printf("Compute: %.6f s\n", comp_max);
        printf("Write:   %.6f s\n", tW1 - tW0);
        fflush(stdout);
    }

    /* cleanup */
    if (rank == 0) {
        free(img);
        free(out_full);
    }
    free(k);

    if (local_in)  free(local_in);
    if (local_out) free(local_out);

    MPI_Type_free(&MPI_PIXEL); /* libero il datatype MPI creato */
    MPI_Finalize();            /* chiusura MPI */
    return 0;
}
