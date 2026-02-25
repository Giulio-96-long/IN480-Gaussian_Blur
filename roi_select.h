#ifndef ROI_SELECT_H
#define ROI_SELECT_H

#ifdef __cplusplus
extern "C" {
#endif

// Ritorna 1 se confermata, 0 se annullata/errore
int roi_select_interactive(const char *image_path,
                           int *out_x, int *out_y, int *out_w, int *out_h);

#ifdef __cplusplus
}
#endif

#endif
