## ============================================================
##  Microbiome Relative Abundance Heatmap
##  Three functions:
##    1. plot_heatmap_all()   – all species (relative abundance)
##    2. plot_heatmap_top()   – top-N species (relative abundance)
##    3. plot_heatmap_zscore()– z-score standardised abundance
##
##  All functions accept a `viridis_palette` argument:
##    "plasma", "viridis", "magma", "inferno", "cividis", "mako",
##    "rocket", "turbo"
## ============================================================

# ── Dependencies ──────────────────────────────────────────────
required_packages <- c(
  "tidyverse",   # data wrangling
  "pheatmap",    # heatmap with clustering
  "RColorBrewer",# colour palettes
  "viridis",     # perceptually-uniform colours
  "grid",        # grob manipulation
  "scales",       # label formatting
  "knitr",     
  "kableExtra",
  "gt",
  "Hmisc"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(pheatmap)
  library(RColorBrewer)
  library(viridis)
  library(grid)
  library(scales)
  library(knitr)
  library(kableExtra)
  library(gt)
  library(Hmisc)
})


## ============================================================
##  Helper: load & pre-process the CSV
##  Returns a named list:
##    $rel_abund  – sample × species relative-abundance matrix
##    $meta       – sample metadata (subCST, Pct columns)
## ============================================================
load_microbiome_csv <- function(filepath,
                                sample_col   = "sample_ID",
                                total_col    = "total_reads",
                                subcst_col   = "subCST",
                                pct_cols     = c("Pct CST0","Pct CST1","Pct CST2")) {

  df <- read.csv(filepath, check.names = FALSE, stringsAsFactors = FALSE)

  # ── Identify taxa columns (everything that's not metadata) ──
  meta_cols <- c(sample_col, total_col, subcst_col, pct_cols)
  taxa_cols <- setdiff(colnames(df), meta_cols)
  if (length(taxa_cols) == 0) stop("No taxa columns found. Check column names.")

  # ── Relative abundance (row-normalised to sum = 1) ──────────
  counts    <- df[, taxa_cols, drop = FALSE]
  counts    <- apply(counts, 2, as.numeric)
  row_sums  <- rowSums(counts, na.rm = TRUE)
  row_sums[row_sums == 0] <- NA
  rel_abund <- sweep(counts, 1, row_sums, "/")
  rownames(rel_abund) <- df[[sample_col]]

  # ── Metadata ─────────────────────────────────────────────────
  meta <- df[, intersect(meta_cols, colnames(df)), drop = FALSE]
  rownames(meta) <- df[[sample_col]]
  meta <- meta[, setdiff(colnames(meta), sample_col), drop = FALSE]
  meta[[subcst_col]] <- as.factor(meta[[subcst_col]])

  # ── Sequencing effort: z-score of total read count ──────────
  # Stored as "Seq. Effort (z)" so it appears as a continuous
  # annotation bar above the heatmap alongside subCST / Pct bars.
  if (total_col %in% colnames(df)) {
    rc   <- as.numeric(df[[total_col]])
    rc_z <- (rc - mean(rc, na.rm = TRUE)) / sd(rc, na.rm = TRUE)
    meta[["Seq. Effort (z)"]] <- rc_z
  } else {
    warning(sprintf("Column '%s' not found; sequencing effort bar skipped.", total_col))
  }

  list(rel_abund = rel_abund, meta = meta,
       subcst_col = subcst_col, pct_cols = pct_cols,
       total_col  = total_col)
}


## ============================================================
##  Core heatmap engine (internal helper)
## ============================================================

# Valid viridis palette names for validation
.VIRIDIS_PALETTES <- c("plasma","viridis","magma","inferno",
                        "cividis","mako","rocket","turbo")

.draw_heatmap <- function(mat,                        # sample × species matrix
                           meta,                       # metadata data.frame
                           subcst_col,                 # column name for subCST
                           title           = "Relative Abundance",
                           save_pdf        = FALSE,    # TRUE → write PDF to disk
                           filename        = "heatmap.pdf",
                           width           = 14,
                           height          = 10,
                           fontsize_row    = 7,
                           fontsize_col    = 9,
                           show_rownames   = TRUE,
                           viridis_palette = "plasma",
                           diverging       = FALSE,
                           zlim            = NULL) {

  # ── Validate palette ─────────────────────────────────────────
  viridis_palette <- match.arg(viridis_palette, .VIRIDIS_PALETTES)

  # ── Transpose so species = rows, samples = columns ──────────
  plot_mat <- t(mat)

  # ── Order samples: subCST first, then Ward's D2 within group ─
  subcst_levels   <- levels(meta[[subcst_col]])
  ordered_samples <- character(0)

  for (lvl in subcst_levels) {
    samp_names <- rownames(meta)[meta[[subcst_col]] == lvl]
    sub_mat    <- t(plot_mat[, samp_names, drop = FALSE])
    if (nrow(sub_mat) > 2) {
      hc        <- hclust(dist(sub_mat, method = "euclidean"), method = "ward.D2")
      sub_order <- samp_names[hc$order]
    } else {
      sub_order <- samp_names
    }
    ordered_samples <- c(ordered_samples, sub_order)
  }

  plot_mat <- plot_mat[, ordered_samples, drop = FALSE]
  meta_ord <- meta[ordered_samples, , drop = FALSE]

  # ── Cluster species (rows) globally ─────────────────────────
  if (nrow(plot_mat) > 2) {
    spc_hc   <- hclust(dist(plot_mat, method = "euclidean"), method = "ward.D2")
    plot_mat <- plot_mat[spc_hc$order, ]
  }

  # ── Heatmap colour palette ───────────────────────────────────
  if (diverging) {
    heat_cols <- colorRampPalette(rev(brewer.pal(11, "RdBu")))(100)
  } else {
    heat_cols <- viridis(100, option = viridis_palette, begin = 0.05, end = 0.97)
  }

  # ── Annotation colours ───────────────────────────────────────
  n_cst   <- length(subcst_levels)
  cst_pal <- setNames(
    brewer.pal(max(3, n_cst), "Set2")[seq_len(n_cst)],
    subcst_levels
  )
  ann_colors <- list()
  ann_colors[[subcst_col]] <- cst_pal

  # Pct columns: sequential blue
  pct_present <- grep("^Pct", colnames(meta_ord), value = TRUE)
  for (pc in pct_present) {
    ann_colors[[pc]] <- colorRampPalette(c("#EFF3FF", "#084594"))(100)
  }

  # Sequencing effort: diverging brown–teal centred on 0
  # Brown (negative z) → white (0) → teal (positive z) makes it
  # immediately clear which samples are under- or over-sequenced
  # relative to the cohort mean.
  if ("Seq. Effort (z)" %in% colnames(meta_ord)) {
    ann_colors[["Seq. Effort (z)"]] <-
      colorRampPalette(c("#8C510A", "#F5F5F5", "#01665E"))(100)
  }

  # ── Column annotation data frame ─────────────────────────────
  ann_cols_use <- c(subcst_col, pct_present,
                    intersect("Seq. Effort (z)", colnames(meta_ord)))
  col_ann <- meta_ord[, ann_cols_use, drop = FALSE]

  # ── Gap positions between subCST groups ─────────────────────
  gap_pos <- cumsum(table(meta_ord[[subcst_col]])[subcst_levels])
  gap_pos <- gap_pos[-length(gap_pos)]

  # ── Colour scale breaks (optional clamping) ──────────────────
  extra_args <- list()
  if (!is.null(zlim)) {
    extra_args$breaks <- seq(zlim[1], zlim[2], length.out = 101)
  }

  # ── Legend labels ────────────────────────────────────────────
  if (!diverging) {
    legend_breaks <- seq(0, 1, by = 0.2)
    legend_labels <- paste0(seq(0, 100, by = 20), "%")
  } else {
    legend_breaks <- NULL
    legend_labels <- NULL
  }

  # ── Assemble pheatmap arguments ──────────────────────────────
  args <- c(list(
    mat               = plot_mat,
    color             = heat_cols,
    annotation_col    = col_ann,
    annotation_colors = ann_colors,
    cluster_rows      = FALSE,
    cluster_cols      = FALSE,
    gaps_col          = gap_pos,
    show_rownames     = show_rownames,
    show_colnames     = TRUE,
    fontsize_row      = fontsize_row,
    fontsize_col      = fontsize_col,
    fontsize          = 9,
    border_color      = NA,
    main              = title,
    angle_col         = 90,
    legend_breaks     = legend_breaks,
    legend_labels     = legend_labels
  ), extra_args)

  # ── Save to PDF or draw to active device ─────────────────────
  if (save_pdf) {
    args$filename <- filename
    args$width    <- width
    args$height   <- height
    message(sprintf("Saving PDF to: %s", filename))
  }

  do.call(pheatmap, args)
  invisible(NULL)
}


## ============================================================
##  FUNCTION 1 – Plot heatmap for ALL species
## ============================================================
#'
#' @param filepath        Path to the CSV file
#' @param sample_col      Column name for sample IDs (default "sample_ID")
#' @param total_col       Column name for total reads (default "total_reads")
#' @param subcst_col      Column name for subCST labels (default "subCST")
#' @param pct_cols        Character vector of Pct column names
#' @param title           Plot title
#' @param save_pdf        If TRUE, write the plot to a PDF file (default FALSE)
#' @param filename        Output file name used when save_pdf = TRUE
#' @param width/height    Figure size in inches (used only when save_pdf = TRUE)
#' @param fontsize_row    Font size for species names
#' @param show_rownames   Show species names (set FALSE if too many)
#' @param viridis_palette One of: "plasma","viridis","magma","inferno",
#'                        "cividis","mako","rocket","turbo"  (default "plasma")
#'
plot_heatmap_all <- function(
    filepath,
    sample_col      = "sample_ID",
    total_col       = "total_reads",
    subcst_col      = "subCST",
    pct_cols        = c("Pct CST0","Pct CST1","Pct CST2"),
    title           = "Relative Abundance \u2013 All Species",
    save_pdf        = FALSE,
    filename        = "heatmap_all.pdf",
    width           = 16,
    height          = 12,
    fontsize_row    = 7,
    show_rownames   = TRUE,
    viridis_palette = "plasma"
) {

  message("Loading data \u2026")
  dat  <- load_microbiome_csv(filepath, sample_col, total_col, subcst_col, pct_cols)
  keep <- colSums(dat$rel_abund, na.rm = TRUE) > 0
  mat  <- dat$rel_abund[, keep, drop = FALSE]
  message(sprintf("Plotting heatmap for %d species \u00d7 %d samples",
                  ncol(mat), nrow(mat)))

  .draw_heatmap(mat, dat$meta, subcst_col,
                title           = title,
                save_pdf        = save_pdf,
                filename        = filename,
                width           = width,
                height          = height,
                fontsize_row    = fontsize_row,
                show_rownames   = show_rownames,
                viridis_palette = viridis_palette)
}


## ============================================================
##  FUNCTION 2 – Plot heatmap for TOP-N species
## ============================================================
#'
#' @param top_n           Number of top species to display (default 20)
#' @param rank_by         How to rank species:
#'                          "mean"       – mean relative abundance
#'                          "prevalence" – fraction of samples where species > 0
#'                          "max"        – maximum relative abundance
#' @param save_pdf        If TRUE, write the plot to a PDF file (default FALSE)
#' @param filename        Output file name used when save_pdf = TRUE
#' @param viridis_palette One of: "plasma","viridis","magma","inferno",
#'                        "cividis","mako","rocket","turbo"  (default "plasma")
#' @param (all other params same as plot_heatmap_all)
#'
plot_heatmap_top <- function(
    filepath,
    top_n           = 20,
    rank_by         = c("mean","prevalence","max"),
    sample_col      = "sample_ID",
    total_col       = "total_reads",
    subcst_col      = "subCST",
    pct_cols        = c("Pct CST0","Pct CST1","Pct CST2"),
    title           = NULL,
    save_pdf        = FALSE,
    filename        = "heatmap_top.pdf",
    width           = 14,
    height          = 10,
    fontsize_row    = 9,
    show_rownames   = TRUE,
    viridis_palette = "plasma"
) {

  rank_by <- match.arg(rank_by)

  message("Loading data \u2026")
  dat      <- load_microbiome_csv(filepath, sample_col, total_col, subcst_col, pct_cols)
  mat_full <- dat$rel_abund

  scores <- switch(rank_by,
    mean       = colMeans(mat_full, na.rm = TRUE),
    prevalence = colMeans(mat_full > 0, na.rm = TRUE),
    max        = apply(mat_full, 2, max, na.rm = TRUE)
  )

  top_species <- names(sort(scores, decreasing = TRUE))[seq_len(min(top_n, length(scores)))]
  mat_top     <- mat_full[, top_species, drop = FALSE]

  if (is.null(title)) {
    title <- sprintf("Relative Abundance \u2013 Top %d Species (ranked by %s)",
                     length(top_species), rank_by)
  }

  message(sprintf("Plotting heatmap for top %d species \u00d7 %d samples",
                  length(top_species), nrow(mat_top)))

  .draw_heatmap(mat_top, dat$meta, subcst_col,
                title           = title,
                save_pdf        = save_pdf,
                filename        = filename,
                width           = width,
                height          = height,
                fontsize_row    = fontsize_row,
                show_rownames   = show_rownames,
                viridis_palette = viridis_palette)
}


## ============================================================
##  FUNCTION 3 – Z-score standardised abundance heatmap
## ============================================================
#'
#' Applies the same z-score transformation used by the microbiome R package:
#' for each species (row), subtract the cross-sample mean and divide by SD.
#' This centres each taxon at 0 and scales by its variability, making rare
#' but variable taxa visible alongside dominant ones.
#'
#' Because values are now diverging (negative = below average,
#' positive = above average), the colour palette automatically switches
#' to a diverging Red-Blue scheme (blue = low, white = 0, red = high),
#' which is the standard for z-score heatmaps. The `viridis_palette`
#' argument is accepted but ignored when diverging = TRUE (the default);
#' set diverging = FALSE to force a viridis palette instead.
#'
#' @param filepath        Path to the CSV file
#' @param top_n           If NULL, use all species. If an integer, first
#'                        subset to the top-N species by mean relative
#'                        abundance, then z-score transform.
#' @param rank_by         Ranking method when top_n is set:
#'                        "mean", "prevalence", or "max"
#' @param sample_col      Column name for sample IDs (default "sample_ID")
#' @param total_col       Column name for total reads (default "total_reads")
#' @param subcst_col      Column name for subCST labels (default "subCST")
#' @param pct_cols        Character vector of Pct column names
#' @param title           Plot title (auto-generated if NULL)
#' @param save_pdf        If TRUE, write the plot to a PDF file (default FALSE)
#' @param filename        Output file name used when save_pdf = TRUE
#' @param width/height    Figure size in inches
#' @param fontsize_row    Font size for species names
#' @param show_rownames   Show species names
#' @param zlim            Symmetric colour scale limits, e.g. c(-3, 3).
#'                        Values outside this range are clamped to the
#'                        extremes of the colour scale. NULL = auto.
#' @param diverging       Use diverging RdBu palette (TRUE, recommended)
#'                        or a viridis palette (FALSE)
#' @param viridis_palette Viridis palette to use when diverging = FALSE
#'
plot_heatmap_zscore <- function(
    filepath,
    top_n           = NULL,
    rank_by         = c("mean","prevalence","max"),
    sample_col      = "sample_ID",
    total_col       = "total_reads",
    subcst_col      = "subCST",
    pct_cols        = c("Pct CST0","Pct CST1","Pct CST2"),
    title           = NULL,
    save_pdf        = FALSE,
    filename        = "heatmap_zscore.pdf",
    width           = 14,
    height          = 10,
    fontsize_row    = 9,
    show_rownames   = TRUE,
    zlim            = c(-3, 3),
    diverging       = TRUE,
    viridis_palette = "mako"
) {

  rank_by <- match.arg(rank_by)

  message("Loading data \u2026")
  dat      <- load_microbiome_csv(filepath, sample_col, total_col, subcst_col, pct_cols)
  mat_full <- dat$rel_abund

  # ── Optional: subset to top-N before z-scoring ──────────────
  if (!is.null(top_n)) {
    scores <- switch(rank_by,
      mean       = colMeans(mat_full, na.rm = TRUE),
      prevalence = colMeans(mat_full > 0, na.rm = TRUE),
      max        = apply(mat_full, 2, max, na.rm = TRUE)
    )
    top_species <- names(sort(scores, decreasing = TRUE))[seq_len(min(top_n, length(scores)))]
    mat_full    <- mat_full[, top_species, drop = FALSE]
    n_label     <- sprintf("Top %d Species", length(top_species))
  } else {
    keep     <- colSums(mat_full, na.rm = TRUE) > 0
    mat_full <- mat_full[, keep, drop = FALSE]
    n_label  <- sprintf("All %d Species", ncol(mat_full))
  }

  # ── Z-score transformation (per species / column) ───────────
  col_means <- colMeans(mat_full, na.rm = TRUE)
  col_sds   <- apply(mat_full, 2, sd, na.rm = TRUE)

  zero_var <- col_sds == 0 | is.na(col_sds)
  if (any(zero_var)) {
    message(sprintf("Dropping %d zero-variance species.", sum(zero_var)))
    mat_full  <- mat_full[, !zero_var, drop = FALSE]
    col_means <- col_means[!zero_var]
    col_sds   <- col_sds[!zero_var]
  }

  mat_z <- sweep(mat_full, 2, col_means, "-")
  mat_z <- sweep(mat_z,   2, col_sds,   "/")

  if (is.null(title))
    title <- sprintf("Z-score Standardised Abundance \u2013 %s", n_label)

  message(sprintf("Plotting z-score heatmap for %d species \u00d7 %d samples",
                  ncol(mat_z), nrow(mat_z)))

  .draw_heatmap(mat_z, dat$meta, subcst_col,
                title           = title,
                save_pdf        = save_pdf,
                filename        = filename,
                width           = width,
                height          = height,
                fontsize_row    = fontsize_row,
                show_rownames   = show_rownames,
                viridis_palette = viridis_palette,
                diverging       = diverging,
                zlim            = zlim)
}


## ============================================================
##  EXAMPLE USAGE (uncomment and edit path to run)
## ============================================================

# MY_FILE <- "path/to/your/data.csv"

# ── Plot to screen (default) ─────────────────────────────────
# plot_heatmap_all(
#   filepath   = MY_FILE,
#   sample_col = "sampleID",
#   total_col  = "read_count"
# )

# ── Save to PDF ───────────────────────────────────────────────
# plot_heatmap_all(
#   filepath        = MY_FILE,
#   sample_col      = "sampleID",
#   total_col       = "read_count",
#   save_pdf        = TRUE,
#   filename        = "heatmap_all.pdf",
#   width           = 18,
#   height          = 14,
#   viridis_palette = "plasma"
# )

# ── Top 30 species to screen ──────────────────────────────────
# plot_heatmap_top(
#   filepath        = MY_FILE,
#   sample_col      = "sampleID",
#   total_col       = "read_count",
#   top_n           = 30,
#   rank_by         = "mean",
#   viridis_palette = "viridis"
# )

# ── Z-score: top 40, save PDF ────────────────────────────────
# plot_heatmap_zscore(
#   filepath   = MY_FILE,
#   sample_col = "sampleID",
#   total_col  = "read_count",
#   top_n      = 40,
#   zlim       = c(-3, 3),
#   save_pdf   = TRUE,
#   filename   = "heatmap_zscore_top40.pdf"
# )

# ── Z-score: all species, viridis instead of RdBu, to screen ─
# plot_heatmap_zscore(
#   filepath        = MY_FILE,
#   sample_col      = "sampleID",
#   total_col       = "read_count",
#   diverging       = FALSE,
#   viridis_palette = "mako",
#   zlim            = NULL,
#   show_rownames   = FALSE
# )



#############################
# Use of scripts to generate heatmap for our oral data
#############################





plot_heatmap_zscore(
   filepath   = "Hyuhn_table_filtered.csv",
   sample_col = "sampleID",
   total_col  = "read_count",
   top_n      = 40,
   zlim       = c(-3, 3),
   save_pdf   = TRUE,
   filename   = "heatmap_zscore_top40.pdf"
 )


############################
# Estimation of Shannon Index for probability of assignment to different CST types 
# and the assessment of a correlation (or lack thereof) betweeen sequencing effort (number of reads per sample) and evenness in the probability of assignment (increased uncertainty)
############################

H <- log(df$Pct.CST0)*df$Pct.CST0 + log(df$Pct.CST1)*df$Pct.CST1 + log(df$Pct.CST2)*df$Pct.CST2
H_norm <- H/log(3)
assesment <- data.frame(cbind(scale(df$read_count, center=TRUE, scale=TRUE),H_norm,H))
colnames(assesment)[1] <- "Sequencing_effort"
ggplot(assesment,aes(x= Sequencing_effort,y=H_norm)) + geom_point()

res_hmisc <- rcorr(as.matrix(assesment), type = "pearson")

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    Row = rownames(cormat)[row(cormat)[ut]],
    Column = rownames(cormat)[col(cormat)[ut]],
    Corr = cormat[ut],
    P = pmat[ut]
  )
}
flat_res <- flattenCorrMatrix(res_hmisc$r, res_hmisc$P)

flat_res$Corr <- round(flat_res$Corr, 2)
flat_res$P <- round(flat_res$P, 4)

kable(flat_res, caption = "Correlation Matrix with P-values") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))


