# IN480 – Gaussian Blur (OpenMP + MPI) con ROI

## Descrizione del Progetto

Progetto sviluppato per il corso **IN480 – Calcolo Parallelo e Distribuito**.

Il progetto implementa un **Gaussian Blur separabile** applicato a immagini in formato **PPM**, 
con supporto alla selezione di una **Regione di Interesse (ROI)**.

Il filtro Gaussiano viene implementato sfruttando la proprietà di separabilità:

1. Passata orizzontale  
2. Passata verticale  

Questo approccio riduce la complessità computazionale rispetto alla convoluzione 2D diretta.

Sono state sviluppate due versioni parallele:

- **OpenMP** (memoria condivisa)
- **MPI** (memoria distribuita)

---

## Funzionalità

- Applicazione del Gaussian Blur **solo sulla ROI**
- Supporto a due modalità di selezione:

### `--region`

ROI specificata da riga di comando: x y w h

### `--select`

Selezione interattiva tramite mouse (SDL2)

- Supporta immagini PPM / PNG / JPG  
- Nella versione MPI solo il **rank 0** apre la finestra grafica  

---

## Contenuto della Repository

- gaussian_blur_omp.c → Versione OpenMP  
- gaussian_blur_mpi.c → Versione MPI  
- roi_select.c, roi_select.h → Gestione ROI  
- Insert_mouse.c → Supporto selezione con mouse  
- stb_image.h, stb_impl.c → Libreria caricamento PNG/JPG  

---

## Requisiti

### OpenMP
- GCC con supporto OpenMP  

### MPI
- OpenMPI oppure MPICH  

### Selezione interattiva
- SDL2  

---

## Installazione dipendenze (Ubuntu/Debian)

```bash
sudo apt install build-essential mpi-default-bin mpi-default-dev libsdl2-dev
```

---

# Compilazione

## Versione OpenMP

```bash
gcc -std=c99 -Wall -Wextra -fopenmp -DHAVE_SDL2 gaussian_blur_omp.c roi_select.c stb_impl.c -o gaussian_blur_omp $(sdl2-config --cflags --libs) -lm
```

## Versione MPI

```bash
mpicc -std=c99 -Wall -Wextra -DHAVE_SDL2 gaussian_blur_mpi.c roi_select.c stb_impl.c -o gaussian_blur_mpi $(sdl2-config --cflags --libs) -lm
```

---

# Esecuzione

## OpenMP

### ROI da riga di comando

```bash
OMP_NUM_THREADS=8 ./gaussian_blur_omp --region input.ppm output.ppm x y w h
```

### ROI selezionata con il mouse

```bash
OMP_NUM_THREADS=8 ./gaussian_blur_omp --select input.ppm output.ppm
```

---

## MPI

### ROI da riga di comando

```bash
mpirun -np 4 ./gaussian_blur_mpi --region input.ppm output.ppm x y w h
```

### ROI selezionata con il mouse

```bash
mpirun -np 4 ./gaussian_blur_mpi --select input.ppm output.ppm
```

(Nella modalità --select solo il rank 0 apre la finestra SDL2.)

---

## Output dei Tempi

Il programma stampa:

- Read  
- Compute  
- Write  
